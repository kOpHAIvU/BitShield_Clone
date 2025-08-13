#include <cstdio>
#include <cstring>

#include "c_runtime_api.h"
#include "adler32.h"

/*!
 * \brief Signature for backend functions exported as DLL.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param out_ret_value The output value of the the return value.
 * \param out_ret_tcode The output type code of the return value.
 * \param resource_handle Pointer to associated resource.
 *
 * \return 0 if success, -1 if failure happens, set error via TVMAPISetLastError.
 */
typedef int (*TVMBackendPackedCFunc)(TVMValue* args, int* type_codes, int num_args,
		TVMValue* out_ret_value, int* out_ret_tcode,
		void* resource_handle);

static uint64_t get_nelems(DLTensor* t) {
	int64_t size = 1;
	for (int i = 0; i < t->ndim; ++i) {
		size *= t->shape[i];
	}
	return size;
}

static uint64_t get_nbytes(DLTensor* t) {
	// Note the () on the RHS. They're very important!
	return get_nelems(t) * ((t->dtype.bits * t->dtype.lanes + 7) / 8);
}

static void copy_tensor(DLTensor* dst, DLTensor* src) {
	uint64_t data_size = get_nbytes(src);
	memcpy((char *) dst->data + dst->byte_offset, (char *) src->data + src->byte_offset, data_size);
}

template <typename T>
static T* data_ptr(DLTensor* t) {
	return (T*) ((char *) t->data + t->byte_offset);
}

typedef union {
	uint32_t as_uint32;
	float as_float;
} punned_float;

static inline bool IsContiguous(const DLTensor& arr) {
	if (arr.strides == nullptr) return true;
	int64_t expected_stride = 1;
	for (int32_t i = arr.ndim; i != 0; --i) {
		int32_t k = i - 1;
		if (arr.strides[k] != expected_stride) return false;
		expected_stride *= arr.shape[k];
	}
	return true;
}

extern "C" int unmask_weights(TVMValue* args, int* type_codes, int num_args,
		TVMValue* out_ret_value, int* out_ret_tcode,
		void* resource_handle) {

	static_assert(sizeof(float) == sizeof(uint32_t), "punned_float won't work");

	DLTensor* in = (DLTensor*) args[0].v_handle;
	DLTensor* out = (DLTensor*) args[1].v_handle;
	int32_t code_offset = (int32_t) args[2].v_int64;
	int32_t code_len = (int32_t) args[3].v_int64;

	// printf("contiguous: %d, %d\n", IsContiguous(*in), IsContiguous(*out));

	// TODO: We should avoid instrumenting layers we don't want at all, so we
	// don't need to copy unchanged data around.
	if (code_len == 0) {
		copy_tensor(out, in);
		return 0;
	}

	// Use asm to get the code by RIP-relative addressing (AT&T syntax)
	uint8_t* rip = nullptr;
	asm volatile(
		"lea (%%rip), %0"
		: "=r"(rip)
	);
	uint8_t* code = (uint8_t*) rip + code_offset;

	auto cksum = adler32(1, code, code_len);
	// printf("code_offset: %d, code_len: %d, cksum: %x\n", code_offset, code_len, cksum);

	for (int64_t i = 0; i < get_nelems(in); ++i) {
		// Do some casting dance
		auto unmasked = data_ptr<punned_float>(in)[i].as_uint32 ^ cksum;
		data_ptr<float>(out)[i] = punned_float{unmasked}.as_float;
	}

	return 0;
}

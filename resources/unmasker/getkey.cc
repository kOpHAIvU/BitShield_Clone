#include <stdint.h>
#include <stdio.h>
#include "adler32.h"

uint64_t ground_truth_magic = 0x1DDECC41629D2D4B;

// Returns a ground truth value for the caller to use.
// The ground_truth_magic is 64 bits to reduce the chance of a collision.
extern "C" uint32_t getkey() {
	auto ret = (uint32_t) ground_truth_magic;
	// printf("getkey() -> %x\n", ret);
	return ret;
}

// extern "C" uint32_t getkey(int32_t code_offset, uint32_t code_len) {

	// if (code_len == 0)
		// return 0;

	// // Use asm to get the code by RIP-relative addressing (AT&T syntax)
	// uint8_t* rip = nullptr;
	// asm volatile(
			// "lea (%%rip), %0"
			// : "=r"(rip)
			// );
	// uint8_t* code = (uint8_t*) rip + code_offset;

	// auto ret = adler32(1, code, code_len);
	// // printf("%d bytes at %p -> %x\n", code_len, code, ret);
	// return ret;
// }

#! /usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GHIDRA_DIR="$SCRIPT_DIR"/ghidra-app
GHIDRA_EXT_DIR="$GHIDRA_DIR"/Ghidra/Extensions
GRADLE_DIR="$SCRIPT_DIR"/gradle-7.6
JAVA17_DIR="$SCRIPT_DIR"/java-17

cd "$SCRIPT_DIR"

# Required:
# sudo apt install -y openjdk-17-jdk-headless unzip (if you have sudo)
# Or use portable Java 17 (no sudo needed)

# Find Java 17 installation
find_java17() {
	# Try multiple common locations
	for jdk_path in \
		"/usr/lib/jvm/java-17-openjdk-amd64" \
		"/usr/lib/jvm/java-17-openjdk" \
		"/usr/lib/jvm/java-17" \
		"/opt/java/17" \
		"$JAVA17_DIR" \
		; do
		if [ -d "$jdk_path" ] && [ -f "$jdk_path/bin/java" ]; then
			# Verify it's Java 17
			JAVA_VER=$("$jdk_path/bin/java" -version 2>&1 | head -n 1 | grep -oE 'version "[0-9]+' | grep -oE '[0-9]+' || echo "0")
			if [ "$JAVA_VER" -eq 17 ] || [ "$JAVA_VER" -eq 0 ]; then
				echo "$jdk_path"
				return 0
			fi
		fi
	done
	
	# Try update-alternatives
	if command -v update-alternatives &> /dev/null; then
		ALT_JAVA=$(update-alternatives --list java 2>/dev/null | grep -i "java-17" | head -1)
		if [ -n "$ALT_JAVA" ]; then
			JAVA17_DIR_ALT=$(dirname "$(dirname "$ALT_JAVA")")
			if [ -d "$JAVA17_DIR_ALT" ]; then
				echo "$JAVA17_DIR_ALT"
				return 0
			fi
		fi
	fi
	
	return 1
}

# Download and install portable Java 17 (no sudo needed)
install_java17_portable() {
	echo "Downloading portable Java 17 (this may take a few minutes)..."
	
	# Detect architecture
	ARCH=$(uname -m)
	if [ "$ARCH" = "x86_64" ]; then
		ARCH="x64"
	elif [ "$ARCH" = "aarch64" ]; then
		ARCH="aarch64"
	else
		echo "Error: Unsupported architecture: $ARCH" >&2
		return 1
	fi
	
	# Use a more reliable URL - Adoptium Temurin 17
	# Try latest LTS version
	JAVA17_URL="https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.13%2B11/OpenJDK17U-jdk_${ARCH}_linux_hotspot_17.0.13_11.tar.gz"
	
	if [ ! -e "$JAVA17_DIR" ] || [ ! -f "$JAVA17_DIR/bin/java" ]; then
		mkdir -p "$JAVA17_DIR"
		cd "$SCRIPT_DIR" || return 1
		
		# Download
		echo "Downloading from: $JAVA17_URL"
		if command -v wget &> /dev/null; then
			if ! wget --progress=bar:force -O java17.tar.gz "$JAVA17_URL" 2>&1; then
				echo "Error: Failed to download Java 17" >&2
				rm -f java17.tar.gz
				return 1
			fi
		elif command -v curl &> /dev/null; then
			if ! curl -L --progress-bar -o java17.tar.gz "$JAVA17_URL"; then
				echo "Error: Failed to download Java 17" >&2
				rm -f java17.tar.gz
				return 1
			fi
		else
			echo "Error: Neither wget nor curl found. Please install one of them." >&2
			return 1
		fi
		
		# Extract
		echo "Extracting Java 17..."
		if ! tar -xzf java17.tar.gz -C "$JAVA17_DIR" --strip-components=1; then
			echo "Error: Failed to extract Java 17" >&2
			rm -f java17.tar.gz
			rm -rf "$JAVA17_DIR"
			return 1
		fi
		rm -f java17.tar.gz
		
		echo "Java 17 installed to: $JAVA17_DIR"
	fi
	
	if [ -f "$JAVA17_DIR/bin/java" ]; then
		echo "$JAVA17_DIR"
		return 0
	fi
	
	echo "Error: Java 17 installation incomplete" >&2
	return 1
}

# Ensure Java 17 is used (Gradle 7.6 requires Java 17 or lower)
if command -v java &> /dev/null; then
	JAVA_VERSION_OUTPUT=$(java -version 2>&1 | head -n 1)
	JAVA_VERSION=$(echo "$JAVA_VERSION_OUTPUT" | grep -oE 'version "[0-9]+' | grep -oE '[0-9]+' || echo "0")
	
	if [ "$JAVA_VERSION" -gt 17 ] || [ "$JAVA_VERSION" -eq 0 ]; then
		echo "Warning: Java $JAVA_VERSION detected. Gradle 7.6 requires Java 17 or lower."
		echo "Attempting to find and use Java 17..."
		
		# Temporarily disable exit on error to allow function to return empty
		set +e
		JAVA17_HOME=$(find_java17 2>/dev/null)
		set -e
		
		if [ -z "$JAVA17_HOME" ]; then
			echo "Java 17 not found in system locations."
			echo "Attempting to install portable Java 17 (no sudo required)..."
			# Temporarily disable exit on error
			set +e
			# Call function and capture only the path (last line of output)
			install_java17_portable > /tmp/java17_install.log 2>&1
			INSTALL_EXIT_CODE=$?
			set -e
			
			# Check if Java 17 was installed successfully
			if [ -f "$JAVA17_DIR/bin/java" ]; then
				JAVA17_HOME="$JAVA17_DIR"
				echo "Java 17 successfully installed to: $JAVA17_HOME"
			elif [ $INSTALL_EXIT_CODE -eq 0 ]; then
				# Try to get path from last line of output
				JAVA17_HOME=$(tail -1 /tmp/java17_install.log | tr -d '\n\r' | grep -E '^/' || echo "")
			fi
			rm -f /tmp/java17_install.log
			
			if [ -z "$JAVA17_HOME" ] || [ ! -f "$JAVA17_HOME/bin/java" ]; then
				echo "Failed to install portable Java 17."
			fi
		else
			echo "Found Java 17 at: $JAVA17_HOME"
		fi
		
		# Trim whitespace and newlines from JAVA17_HOME
		JAVA17_HOME=$(echo "$JAVA17_HOME" | tr -d '\n\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
		
		if [ -n "$JAVA17_HOME" ] && [ -f "$JAVA17_HOME/bin/java" ]; then
			export JAVA_HOME="$JAVA17_HOME"
			export PATH="$JAVA_HOME/bin:$PATH"
			echo "Using Java 17 from: $JAVA_HOME"
			java -version
		else
			echo ""
			echo "Error: Could not find or install Java 17."
			echo "JAVA17_HOME was: '$JAVA17_HOME'"
			if [ -d "$JAVA17_DIR" ]; then
				echo "Checking $JAVA17_DIR..."
				ls -la "$JAVA17_DIR/bin/java" 2>&1 || echo "java binary not found"
			fi
			echo ""
			echo "Please try one of the following:"
			echo "  1. Ask your administrator to install: sudo apt install -y openjdk-17-jdk-headless"
			echo "  2. Download Java 17 manually and set JAVA_HOME:"
			echo "     export JAVA_HOME=/path/to/java-17"
			echo "     export PATH=\$JAVA_HOME/bin:\$PATH"
			echo ""
			exit 1
		fi
	fi
fi

if [ ! -e "$GRADLE_DIR" ]; then
	wget -O gradle.zip \
		https://services.gradle.org/distributions/gradle-7.6-bin.zip
	unzip gradle.zip
	rm -f gradle.zip
fi

if [ ! -e "$GHIDRA_DIR" ]; then
	wget -O ghidra.zip \
		https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.2.2_build/ghidra_10.2.2_PUBLIC_20221115.zip
	unzip ghidra.zip
	mv ghidra_10.*_PUBLIC "$GHIDRA_DIR"
	rm -f ghidra.zip
fi

if [ ! -e "$GHIDRA_EXT_DIR"/ghidrathon ]; then
	# Clone ghidrathon if it doesn't exist
	GHIDRATHON_DIR="$SCRIPT_DIR"/ghidrathon
	if [ ! -e "$GHIDRATHON_DIR" ]; then
		echo "Cloning Ghidrathon repository..."
		git clone https://github.com/mandiant/Ghidrathon.git "$GHIDRATHON_DIR"
	fi
	
	pushd "$GHIDRATHON_DIR"
	echo "Building Ghidrathon extension..."
	if ! "$GRADLE_DIR"/bin/gradle -PGHIDRA_INSTALL_DIR="$GHIDRA_DIR" build; then
		echo ""
		echo "Warning: Failed to build Ghidrathon extension."
		echo "This is usually because the 'jep' (Java Embedded Python) dependency is missing."
		echo ""
		echo "Ghidrathon is only needed if you plan to use CIG (Coverage Integrity Guard)."
		echo "If you're using --cig ncnp (no CIG), you can skip this step."
		echo ""
		echo "To fix this later, you may need to:"
		echo "  1. Install jep Python package: pip install jep"
		echo "  2. Or check ghidrathon's build.gradle for jep dependency configuration"
		echo ""
		echo "Continuing without Ghidrathon extension..."
		popd
	else
		mv dist/*.zip "$GHIDRA_EXT_DIR"
		pushd "$GHIDRA_EXT_DIR"
		unzip ./*.zip
		rm ./*.zip
		popd
		popd
		echo "Ghidrathon extension installed successfully."
	fi
fi

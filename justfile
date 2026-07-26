# Format all C++ sources in place
format:
  uv run --only-group dev clang-format -i include/hyperjet/*.h python/src/*.h python/src/*.cpp test/src/*.cpp benchmark/src/*.cpp

# Generate the compilation database that clang-tidy needs
compile-db:
  #!/usr/bin/env bash
  set -euo pipefail
  # The test build is the right entry point: it instantiates the templates, and
  # an uninstantiated template is never analysed.
  args=(-Stest -Bbuild/tidy -DCMAKE_BUILD_TYPE=Debug
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
  # AppleClang knows the macOS SDK implicitly, the standalone clang-tidy does
  # not, so the sysroot has to end up in the compilation database.
  if [ "$(uname)" = Darwin ]; then
    args+=("-DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)")
  fi
  cmake "${args[@]}" > /dev/null

# One invocation for all units rather than run-clang-tidy.py: with only two of
# them the parallel driver is no faster, and it reports every finding in the
# header once per unit instead of once.

# Run clang-tidy over the test translation units
tidy: compile-db
  uv run --only-group tidy clang-tidy -p build/tidy --quiet test/src/*.cpp

# Same, applying the fixes clang-tidy can make itself
tidy-fix: compile-db
  uv run --only-group tidy clang-tidy -p build/tidy --quiet --fix test/src/*.cpp

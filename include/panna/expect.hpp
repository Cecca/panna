#pragma once

#include <iostream>
#include <cstdlib>

#ifdef EXPECT_ACTIVE
#pragma message("panna: expect-checking is ACTIVE (EXPECT_ACTIVE defined)")
#define expect(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::cerr << "Assertion failed: (" << #condition << ")\n"     \
                      << "File: " << __FILE__ << "\n"                     \
                      << "Line: " << __LINE__ << std::endl;               \
            std::abort();                                                \
        }                                                                \
    } while (0)
#else
// When EXPECT_ACTIVE is undefined, expect compiles to a no-op. The sizeof
// keeps the condition in an unevaluated context, so referenced variables do
// not trigger unused-variable warnings.
#define expect(condition) ((void)sizeof(condition))
#endif

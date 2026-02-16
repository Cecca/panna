#pragma once

#include <chrono>
#include "panna/logging.hpp"

namespace panna {
    class Timer {
        const char * name;
        const std::chrono::steady_clock::time_point start;

    public:
        Timer( const char* name ): name( name ), start( std::chrono::steady_clock::now() ) {
        }

        ~Timer() {
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
            LOG_INFO( "timer-name", name, "elapsed_ms", elapsed, "msg", "stop timer" );
        }
    };
} // namespace panna
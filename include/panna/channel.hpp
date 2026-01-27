#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <condition_variable>

namespace panna {
    /// A communication channel between threads. It has a fixed size, and readers
    /// and writers block waiting for either space or new data elements, if either
    /// is lacking.
    template <typename T>
    struct Channel {
    private:
        /// Does the channel accept new messages?
        bool closed;
        /// How many messages can be in flight at any one time?
        size_t capacity;
        /// The actual messages
        std::deque<T> messages;
        
        std::mutex mutex;
        std::condition_variable cv_slots; 
        std::condition_variable cv_messages;

    public:
        explicit Channel( size_t capacity ):
            closed( false ),
            capacity( capacity )
             {}

        /// Send an items (moving it) and returns `true` if the message has been sent
        bool send( T&& message ) {
            std::unique_lock<std::mutex> lock(mutex);
            cv_slots.wait(lock, [this](){ return messages.size() < capacity || closed; });
            
            if (closed) return false;
            
            messages.push_back(std::move(message));
            cv_messages.notify_one();
            return true;
        }

        std::optional<T> receive(){
            std::unique_lock<std::mutex> lock(mutex);
            cv_messages.wait(lock, [this](){ return !messages.empty() || closed; });

            if (messages.empty()) {
                return std::nullopt;
            }
            T msg = std::move(messages.front());
            messages.pop_front();
            cv_slots.notify_one();
            return msg;
        }

        void close() {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
            cv_slots.notify_all();
            cv_messages.notify_all();
        }
    };

} // namespace panna

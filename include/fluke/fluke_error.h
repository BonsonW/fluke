/* @file error.h
**
** error checking macros/functions and error messages

MIT License

Copyright (c) 2018  Hasindu Gamaarachchi (hasindu@unsw.edu.au)
Copyright (c) 2018  Thomas Daniell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

#ifndef FLUKE_ERROR_H
#define FLUKE_ERROR_H

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define fluke_log_level get_fluke_log_level()

// the level of verbosity in the log printed to the standard error
enum fluke_log_level_opt {
    FLUKE_LOG_OFF,      // nothing at all
    FLUKE_LOG_ERR,      // error messages
    FLUKE_LOG_WARN,     // warning and error messages
    FLUKE_LOG_INFO,     // information, warning and error messages
    FLUKE_LOG_VERB,     // verbose, information, warning and error messages
    FLUKE_LOG_DBUG,     // debugging, verbose, information, warning and error messages
    FLUKE_LOG_TRAC      // tracing, debugging, verbose, information, warning and error messages
};

enum fluke_log_level_opt get_fluke_log_level();
void set_fluke_log_level(enum fluke_log_level_opt level);

#define FLUKE_DEBUG_PREFIX "[FLUKE_DEBUG] %s: " /* TODO function before debug */
#define FLUKE_VERBOSE_PREFIX "[FLUKE_INFO] %s: "
#define FLUKE_INFO_PREFIX "[%s::FLUKE_INFO]\033[1;34m "
#define FLUKE_WARNING_PREFIX "[%s::FLUKE_WARNING]\033[1;33m "
#define FLUKE_ERROR_PREFIX "[%s::FLUKE_ERROR]\033[1;31m "
#define FLUKE_NO_COLOUR "\033[0m"

#define FLUKE_LOG_TRACE(msg, ...) { \
    if (fluke_log_level >= FLUKE_LOG_TRAC) { \
        fprintf(stderr, FLUKE_DEBUG_PREFIX msg \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#define FLUKE_LOG_DEBUG(msg, ...) { \
    if (fluke_log_level >= FLUKE_LOG_DBUG) { \
        fprintf(stderr, FLUKE_DEBUG_PREFIX msg \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#define FLUKE_VERBOSE(msg, ...) { \
    if (fluke_log_level >= FLUKE_LOG_VERB) { \
        fprintf(stderr, FLUKE_VERBOSE_PREFIX msg "\n", __func__, __VA_ARGS__); \
    } \
}

#define FLUKE_INFO(msg, ...) { \
    if (fluke_log_level >= FLUKE_LOG_INFO) { \
        fprintf(stderr, FLUKE_INFO_PREFIX msg FLUKE_NO_COLOUR "\n", __func__, __VA_ARGS__); \
    } \
}

#define FLUKE_WARNING(msg, ...) { \
    if (fluke_log_level >= FLUKE_LOG_WARN) { \
        fprintf(stderr, FLUKE_WARNING_PREFIX msg FLUKE_NO_COLOUR \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#define FLUKE_ERROR(msg, ...) { \
    if (fluke_log_level >= FLUKE_LOG_ERR) { \
        fprintf(stderr, FLUKE_ERROR_PREFIX msg FLUKE_NO_COLOUR \
                " At %s:%d\n", \
                __func__, __VA_ARGS__, __FILE__, __LINE__ - 1); \
    } \
}

#ifdef __cplusplus
}
#endif

#endif

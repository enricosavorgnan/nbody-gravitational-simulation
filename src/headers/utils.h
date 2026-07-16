//
// Created by enric on 16/07/26.
//

#ifndef UTILS_H
#define UTILS_H
#endif //UTILS_H

#include <errno.h>
#include <stderr.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>


#include "./nbody_common.h"


static void die (const char *format, ...);

static size_t parse_size (const char *text, const char *name);
static dtype parse_dtype (const char *text, const char *name);
static const char *option_value (int *i, int argc, char **argv, const char *key);
static void *checked_aligned_alloc (size_t nbytes, size_t alignment);
static void checked_fread (void *ptr, const size_t size, size_t nmemb, FILE *stream, const char *path, const char *what);
static void checked_fwrite (const void *ptr, const size_t size, size_t nmemb, FILE *stream, const char *path, const char *what);
static int compare_doubles (const void *a, const void *b);


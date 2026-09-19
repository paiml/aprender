// PMAT-3091 scalar: print llama.cpp's own system_info (llama_print_system_info -> ggml_cpu_has_* compile-time features)
// for whichever libllama this is linked against. The producer never prints it, so this is the engaged proof.
#include <stdio.h>
const char * llama_print_system_info(void);
int main(void) { printf("%s\n", llama_print_system_info()); return 0; }

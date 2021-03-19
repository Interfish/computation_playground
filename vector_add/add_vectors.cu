

const char *ptxCode = "
    .version 7.0                                               \n
    .target sm_50, texmode_independent                         \n
    .address_size 32                                           \n
                                                               \n
    .entry add_vectors(                                        \n
        .param .u32 .ptr .global .align 4 add_vectors_param_0, \n
        .param .u32 .ptr .global .align 4 add_vectors_param_1, \n
        .param .u32 .ptr .global .align 4 add_vectors_param_2, \n
        .param .u32 add_vectors_param_3                        \n
    ) {                                                        \n
        .reg .pred      %p<2>;                                 \n
        .reg .s32       %r<21>;                                \n
        ld.param.u32    %r9, [add_vectors_param_3];            \n
        mov.u32         %r5, %envreg3;                         \n
        mov.u32         %r6, %ntid.x;                          \n
        mov.u32         %r7, %ctaid.x;                         \n
        mov.u32         %r8, %tid.x;                           \n
        add.s32         %r10, %r8, %r5;                        \n
        mad.lo.s32      %r4, %r7, %r6, %r10;                   \n
        setp.lt.s32     %p1, %r4, %r9;                         \n
        ld.param.u32    %r18, [add_vectors_param_0];           \n
        ld.param.u32    %r19, [add_vectors_param_1];           \n
        ld.param.u32    %r20, [add_vectors_param_2];           \n
        @%p1 bra        BB0_2;                                 \n
                                                               \n
        ret;                                                   \n
                                                               \n
    BB0_2:                                                     \n
        shl.b32         %r11, %r4, 2;                          \n
        add.s32         %r12, %r18, %r11;                      \n
        add.s32         %r13, %r19, %r11;                      \n
        ld.global.u32   %r14, [%r13];                          \n
        ld.global.u32   %r15, [%r12];                          \n
        add.s32         %r16, %r14, %r15;                      \n
        add.s32         %r17, %r20, %r11;                      \n
        st.global.u32   [%r17], %r16;                          \n
        ret;                                                   \n
    }"

int main(int argc, char *argv[]) {
    nvPTXCompilerHandle compiler;
    nvPTXCompilerCreate(&compiler, (size_t)strlen(ptxCode), ptxCode);
    const char* compile_options[] = { "--gpu-name=sm_70", "--verbose" };
    nvPTXCompilerCompile(compiler, 2, compile_options);
}
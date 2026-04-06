// Simple fork test program for Sprint 17
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    printf("Parent PID: %d\n", getpid());

    pid_t pid = fork();

    if (pid == 0) {
        // Child process
        printf("Child PID: %d\n", getpid());
        // Child does a simple syscall
        write(1, "child\n", 6);
        return 0;
    } else if (pid > 0) {
        // Parent process
        printf("Forked child: %d\n", pid);
        // Parent does a syscall
        write(1, "parent\n", 7);
        wait(NULL);
        return 0;
    } else {
        perror("fork");
        return 1;
    }
}

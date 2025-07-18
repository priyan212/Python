#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R return
#define F for
#define P printf

typedef struct { int a, b; } T;

int x[99], y = 7, z = 13;

void f(int *a, int n) {
    int i, j, t;
    F(i = 0; i < n; i++) a[i] = rand() % 100;
    F(i = 0; i < n; i++) F(j = 0; j < n - 1; j++) if (a[j] > a[j + 1]) {
        t = a[j]; a[j] = a[j + 1]; a[j + 1] = t;
    }
}

int g(char *s) {
    int i = 0, r = 0;
    while (s[i]) r += s[i++] * (i % 3 + 1);
    return r % 1000;
}

int h(int a, int b) {
    if (b == 0) return a;
    return h(b, a % b);
}

int q(T t) {
    return (t.a * t.a + t.b * t.b) % 101;
}

int main() {
    char *p = malloc(64), *s = "XyZ123";
    T t = { .a = 17, .b = 23 };
    int i;
    
    strcpy(p, s);
    f(x, y);
    i = g(p);
    y = h(i, q(t));
    
    if (y % 2 == 0) {
        F(i = 0; i < z; i += 2) P("%c", p[i % strlen(p)] ^ (i + 31));
    } else {
        F(i = 1; i < z; i += 3) P("%c", p[i % strlen(p)] ^ (i + 47));
    }

    free(p);
    R 0;
}

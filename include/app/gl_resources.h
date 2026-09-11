#pragma once

#include <GL/glew.h>

void initTextures();
void initVAO(void);
GLuint initShader();
void deletePBO(GLuint* pbo);
void deleteTexture(GLuint* tex);
void cleanupCuda();
void initCuda();
void initPBO();

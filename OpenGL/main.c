#include <GL/glut.h>

// 定义立方体的顶点
GLfloat vertices[][3] = {
    {-0.5, -0.5, -0.5},
    {-0.5, -0.5, 0.5},
    {-0.5, 0.5, -0.5},
    {-0.5, 0.5, 0.5},
    {0.5, -0.5, -0.5},
    {0.5, -0.5, 0.5},
    {0.5, 0.5, -0.5},
    {0.5, 0.5, 0.5}
};

// 定义立方体的边
int edges[][2] = {
    {0, 1}, {1, 3}, {3, 2}, {2, 0}, // 下面
    {4, 5}, {5, 7}, {7, 6}, {6, 4}, // 上面
    {0, 4}, {1, 5}, {2, 6}, {3, 7}  // 侧面
};

// 初始化OpenGL环境
void init() {
    glClearColor(0.0, 0.0, 0.0, 1.0); // 设置背景颜色为黑色
    glEnable(GL_DEPTH_TEST); // 启用深度测试
}

// 绘制立方体
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清除颜色和深度缓冲区
    glLoadIdentity(); // 重置模型视图矩阵

    // 将立方体移动到原点，并绕y轴旋转
    glTranslatef(0.0, 0.0, -5.0);
    glRotatef(20.0, 1.0, 1.0, 1.0);

    // 绘制立方体的边
    glBegin(GL_LINES);
    for (int i = 0; i < 12; i++) {
        int v1 = edges[i][0];
        int v2 = edges[i][1];
        glVertex3fv(vertices[v1]);
        glVertex3fv(vertices[v2]);
    }
    glEnd();

    glutSwapBuffers(); // 交换前后缓冲区
}

// 窗口大小改变时的处理函数
void reshape(int w, int h) {
    glViewport(0, 0, w, h); // 设置视口大小
    glMatrixMode(GL_PROJECTION); // 切换到投影矩阵
    glLoadIdentity(); // 重置投影矩阵
    gluPerspective(45.0, (GLfloat)w / (GLfloat)h, 0.1, 100.0); // 设置透视投影
    glMatrixMode(GL_MODELVIEW); // 切换回模型视图矩阵
    glLoadIdentity(); // 重置模型视图矩阵
}

int main(int argc, char** argv) {
    glutInit(&argc, argv); // 初始化GLUT
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // 设置显示模式
    glutInitWindowSize(640, 480); // 设置窗口大小
    glutCreateWindow("Rotating Cube"); // 创建窗口
    init(); // 初始化OpenGL环境
    glutDisplayFunc(display); // 设置显示回调函数
    glutReshapeFunc(reshape); // 设置窗口大小改变回调函数
    glutMainLoop(); // 进入GLUT主循环
    return 0;
}
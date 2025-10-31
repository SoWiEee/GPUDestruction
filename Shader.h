#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>

class Shader {
public:
    // �ڭ̪� Shader Program ID
    unsigned int ID;

    // �غc�l�GŪ���ëإ� Shader
    Shader(const char* vertexPath, const char* fragmentPath);

    // �ҥ� Shader
    void use();

    // Utility uniform �禡
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setMat4(const std::string& name, const glm::mat4& mat) const;

private:
    // �ˬd�sĶ/�s�����~
    void checkCompileErrors(unsigned int shader, std::string type);
};
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>

class Shader {
public:
    // 我們的 Shader Program ID
    unsigned int ID;

    // 建構子：讀取並建立 Shader
    Shader(const char* vertexPath, const char* fragmentPath);

    // 啟用 Shader
    void use();

    // Utility uniform 函式
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setMat4(const std::string& name, const glm::mat4& mat) const;

private:
    // 檢查編譯/連結錯誤
    void checkCompileErrors(unsigned int shader, std::string type);
};
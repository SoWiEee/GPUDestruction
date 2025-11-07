#version 450 core

in vec3 ourColor;
in vec2 TexCoord;
in vec3 FragPos;
in vec4 FragPosLightSpace;

out vec4 FragColor;

uniform sampler2D ourTexture; // 石塊紋理
uniform sampler2D shadowMap;  // 陰影貼圖

// 光源位置 (用於計算光照)
uniform vec3 lightPos;
uniform vec3 viewPos; // 攝影機位置

// 陰影計算函式
float CalculateShadow(vec4 fragPosLightSpace) {
    // 1. 執行透視除法 (將座標從 -1,1 轉換到 0,1)
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // 2. 取得目前像素的深度 (從光源視角)
    float currentDepth = projCoords.z;
    
    // 3. 取得陰影貼圖中儲存的「最近深度」
    float closestDepth = texture(shadowMap, projCoords.xy).r; 

    // 加上一個 Bias，防止 shadow acne (物體自己對自己產生陰影)
    float bias = 0.005;
    
    // 如果目前深度 > 最近深度 + 偏移，代表在陰影中
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    
    // (防止超出邊界)
    if(projCoords.z > 1.0)
        shadow = 0.0;
        
    return shadow;
}

void main() {
    // 取得基礎紋理顏色
    vec4 texColor = texture(ourTexture, TexCoord) * vec4(ourColor, 1.0);
    
    // 計算基礎光照 (Blinn-Phong)
    vec3 ambient = 0.3 * texColor.rgb; // 環境光
    
    vec3 norm = vec3(0.0, 1.0, 0.0); // 假設方塊是平的，先不用法線貼圖
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = (diff * 0.7) * texColor.rgb; // 漫射光
    
    // (先不加鏡面光，保持簡單)
    
    // 計算陰影
    float shadow = CalculateShadow(FragPosLightSpace);
    
    // 組合最終顏色 (環境光不受陰影影響，但漫射光會)
    vec3 lighting = ambient + (1.0 - shadow) * diffuse;
    FragColor = vec4(lighting, 1.0);
}
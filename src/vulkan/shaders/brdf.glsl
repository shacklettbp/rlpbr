#define M_PI (3.1415926535897932)
#define M_1_PI (0.3183098861837907)

struct BRDFParams {
    vec3 Li;
    vec3 toLight;
    vec3 L;
    vec3 toView;
    vec3 V;
    vec3 N;
    float NdV;
    float NdL;
    vec3 H;
};

BRDFParams makeBRDFParams(vec3 light_pos, vec3 fragment_pos,
                          vec3 normal, vec3 light_color)
{
    BRDFParams params;
    params.Li = light_color;
    params.toLight = light_pos - fragment_pos;
    params.L = normalize(params.toLight);
    params.toView = -fragment_pos;
    params.V = normalize(params.toView);
    params.N = normalize(normal);
    params.NdV = clamp(dot(params.N, params.V), 0.f, 1.f);
    params.NdL = dot(params.N, params.L);
    params.H = normalize(params.L + params.V);

    return params;
}

vec3 blinnPhong(BRDFParams bp, float shininess,
                vec3 base_diffuse, vec3 base_specular)
{
    if (bp.NdL < 0) return vec3(0.0);

    vec3 diffuse = M_1_PI * bp.Li * base_diffuse;

    vec3 specular = base_specular * pow(max(dot(bp.N, bp.H), 0.f), shininess);

    return bp.NdL * (diffuse + specular);
}

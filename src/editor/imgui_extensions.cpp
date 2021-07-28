#include "imgui_extensions.hpp"

#include <imgui_internal.h>

#include <iostream>

using namespace ImGui;

namespace ImGuiEXT {

bool DragScalarNSeparateRange(const char* label,
                              ImGuiDataType data_type,
                              void* p_data,
                              int components,
                              float *v_speed,
                              const void* p_min,
                              const void* p_max,
                              const char* format,
                              ImGuiSliderFlags flags)
{
    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImGuiContext& g = *GImGui;
    bool value_changed = false;
    BeginGroup();
    PushID(label);
    PushMultiItemsWidths(components, CalcItemWidth());
    size_t type_size = DataTypeGetInfo(data_type)->Size;
    for (int i = 0; i < components; i++)
    {
        PushID(i);
        if (i > 0)
            SameLine(0, g.Style.ItemInnerSpacing.x);

        value_changed |= DragScalar("", data_type, p_data, v_speed[i], p_min, p_max, format, flags);
        PopID();
        PopItemWidth();
        p_data = (void*)((char*)p_data + type_size);
        p_min = (void*)((char*)p_min + type_size);
        p_max = (void*)((char*)p_max + type_size);
    }
    PopID();

    const char* label_end = FindRenderedTextEnd(label);
    if (label != label_end)
    {
        SameLine(0, g.Style.ItemInnerSpacing.x);
        TextEx(label, label_end);
    }

    EndGroup();
    return value_changed;
}

bool DragFloat3SeparateRange(const char* label,
                             float* p_data,
                             float *v_speed,
                             const float* p_min,
                             const float* p_max,
                             const char* format,
                             ImGuiSliderFlags flags)
{
    return DragScalarNSeparateRange(label, ImGuiDataType_Float, p_data, 3,
                                    v_speed, p_min, p_max, format, flags);
}

void PushDisabled(bool disabled)
{
    return ImGui::PushDisabled(disabled);
}

void PopDisabled()
{
    return ImGui::PopDisabled();
}

}

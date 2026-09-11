#include "pathtrace/pathtrace_state.h"

void InitDataContainer(GuiDataContainer* imGuiData)
{
    PathTraceState& pt_state = PathTraceState::Get();
    pt_state.guiData = imGuiData;
}

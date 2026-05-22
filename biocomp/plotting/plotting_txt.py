"""Biology-domain dispatch: drawing-function FQN → its ASCII counterpart."""

from jeanplot.plots.txt import smooth_1d_txt, smooth_2d_txt, smooth_3d_txt

TXT_PLOT_FUNCTION_MAP = {
    "jeanplot.plots.smooth_1d.smooth_1d": smooth_1d_txt,
    "jeanplot.plots.smooth_2d.smooth_2d": smooth_2d_txt,
    "jeanplot.plots.smooth_3d.smooth_3d": smooth_3d_txt,
    "biocomp.plotutils.smooth": None,
}


def get_txt_plot_function(original_func_name: str):
    if original_func_name in TXT_PLOT_FUNCTION_MAP:
        return TXT_PLOT_FUNCTION_MAP[original_func_name]
    for key, func in TXT_PLOT_FUNCTION_MAP.items():
        if original_func_name.endswith(key.split(".")[-1]):
            return func
    return None


__all__ = ["TXT_PLOT_FUNCTION_MAP", "get_txt_plot_function"]

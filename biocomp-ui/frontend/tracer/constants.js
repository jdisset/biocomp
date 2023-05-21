
const BASE_COLORS = {
  dark_blue: "#0a9396",
  dark_red: "#ae2012",
  light_blue: "#94d2bd",
  orange: "#ee9b00",
  neutral_yellow: "#e9d8a6",
  key: "#001219",
  grey: "#AAAAAA",
};

const COLORS = {
  selected_trace: "#e0000090",
  unselected_trace: BASE_COLORS.grey+ "25",

  unselected_point: BASE_COLORS.grey,
  selected_point: BASE_COLORS.orange,

  in_range: "#777",
  out_of_range: BASE_COLORS.grey + "55",

  min: BASE_COLORS.neutral_yellow,
  max: BASE_COLORS.dark_red,
};

// export the constants
export { COLORS };

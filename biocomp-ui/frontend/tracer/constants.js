
const BASE_COLORS = {
  dark_blue: "#0a9396",
  dark_red: "#ae2012",
  light_blue: "#94d2bd",
  orange: "#ee9b00",
  neutral_yellow: "#e9d8a6",
  key: "#001219",
  grey: "#555555",
};

const COLORS = {
  selected_trace: "#ffca3a80",
  unselected_trace: BASE_COLORS.grey+ "14",

  unselected_point: BASE_COLORS.grey,
  selected_point: BASE_COLORS.orange,

  in_range: "#777",
  out_of_range: BASE_COLORS.grey + "33",

  min: BASE_COLORS.neutral_yellow,
  max: BASE_COLORS.dark_red,

  transcription: '#1982c4',
  translation: '#6a4c93',
  output: '#ffca3a',
  input: '#ffca3a',
  sequestron_ERN: '#ff595e',
  source : '#8ac926',
  aggregation: '#fb8b24',
  aggregation: '#222',
  inv_aggregation: '#bbb',
  inv_source: '#bbb',
  inv_translation: '#bbb',
  inv_transcription: '#bbb',
};

// export the constants
export { COLORS };


var colors = [
  "#eb50c7",
  "#71db4f",
  "#bf6ceb",
  "#cbd83b",
  "#8389e4",
  "#72a84a",
  "#ed5886",
  "#68de99",
  "#ef5a40",
  "#6fdcd9",
  "#df873c",
  "#6da5dc",
  "#c5a444",
  "#d585c2",
  "#d4de7d",
  "#c7a6c2",
  "#62a989",
  "#d58379",
  "#c2dfb0",
  "#8bb3c3",
  "#e4c1a3",
  "#a19572",
];

var seed = 1;
function random() {
  var x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

function recomputeAllRules(e) {
  var sheet = SpreadsheetApp.getActive().getSheetByName("parts");
  if (sheet.getSheetId() != e.source.getActiveSheet().getSheetId()) {
    return;
  }
  var range = sheet.getRange("B2:B");
  var uniqueNames = range
    .getValues()
    .filter(function (value) {
      return value != "";
    })
    .map(function (value) {
      return value[0];
    })
    .filter(function (value, index, self) {
      return self.indexOf(value) === index;
    }); 
  uniqueNames.sort();
  var rules = [];
  sheet.clearConditionalFormatRules();
  for (var i = 0; i < uniqueNames.length; i++) {
    if (uniqueNames[i] != "") {
      var rule = SpreadsheetApp.newConditionalFormatRule()
        .setRanges([range])
        .whenTextEqualTo(uniqueNames[i])
        .setBackground(colors[i % colors.length])
        .build();
      rules.push(rule);
    }
  }
  sheet.setConditionalFormatRules(rules);
}

// trigger on open
function onEdit(e) {
  recomputeAllRules(e);
}

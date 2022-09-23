# Useful links
- biocomp-database: 
https://docs.google.com/spreadsheets/d/1K_2bt90E-Wk-A9PYGXGbKDJy-olojKtksy1jxCQAzME/edit?usp=sharing

> *PLEASE MAKE SURE THE FILES ARE VALID JSON5*
- validate json5: https://codebeautify.org/json5-validator
- auto format json5: https://mtp.tools/formatters/json5-formatter

# 3 types of file: recipe, xp, data

## Recipe files
#### Naming convention: `{recipe_name}.recipe.json5`
Store in `recipes` folder

Recipes are json (or json5) files describing a set of interacting constructs.
They can be written manually to describe an experiment in which they have been implemented, or be outputed by the biocompiler 
(in which case they may contain additional information). Each recipe file allows the compiler to build a corresponding compute graph.

They follow this general template:



``` JSON5
{
	name: "L2_pGW0042+CasE-R", // should be a unique name for this recipe
	description: "Aggregation of L2_pGW0010 and NW-B (L1 vs L2), cotransfection ",
	notes: "Manually designed by Jean from Georg's L2 vs L1 coTx study. Meant to be a simple template.", // notes are optional but it would be nice to briefly describe origin and intent.
	content: [
		// list of aggregations (which are lists of cotransfected plasmids)
		{
			// aggregation 0
			sources: [
				{
					ratio: 0.5,
					plasmid: "pNW114.001" // plasmid id, should match either an L1 or an L2 in the database
				},
				{
					ratio: 0.5,
					plasmid: "pAK0007",
					notes: "This is the eBFP marker", // optional notes about this plasmid

					// some biocomp additions (not needed when designed manually for traning purposes)
					optimal_intensity: 152.4, // MEFL intensity at which the circuit is predicted to produce the desired output.
					sensitivity: 0.75 // I'm thinking about a few ways to indicate how critical it is to pick the right intensity
				}
			],
			notes: "Aggregation of CasE and a marker" // optional notes for the whole aggregation object
		},
		{
			// aggregation 1: only one plasmid
			sources: [
				{
					ratio: 1.0, // could be anything really since there's only one plasmid in this aggregation
					plasmid: "pGW0010"

					// + same potential biocomp additions
					// ...
				}
			],
			notes: "Single independant L2" // optional notes for the whole aggregation object
		}
	]
}
```

## XP files
#### Naming convention: `{xpname}.xp.json5`
Store in `experiments/{xpname}/` folder
XP files (aka metadata files) contain information related to the content and the conditions of a specific experiment and its data collection. 
An experiment implements one or several recipes, and its data collection is organised in [tubes/samples/runs ??]. 

Experiment should have unique identifiers, a string with no spaces. I suggest we define a naming pattern and stick to it.
` {date of flow:yyymmdd}-{INITIALS of transfection operator}-{shortdescription:no caps, just like a hashtag}-{optional number}`. Examples:
- `20220708-CVDM-uorfthings`
- `20220501-GW-l1vsl2`

Here is the template for an xp file:

``` json5
{
	name: "20220708-CVDM-uorfthings",
	flow_date: "2022-07-08",
	transfection_date: "2022-07-05",

	samples: [
		// list of tubes / data files
		{
			name: "tube000",
			recipe: "L2_pGW0042+CasE-R",
			notes: "..."
		},
		{
			name: "tube001",
			recipe: "L2_pGW0042+CasE-R",
			notes: "..."
		}
	],

	color_names: {
		// REQUIRED! matches parts that are used in the implemented recipes to a channel name in the data file
		eBFP: "Pacific_Blue_A",
		eYFP: "FITC_A",
		NeonGreen: "FITC_A",
		mKate: "PE_Texas_Red_A",
		iRFP: "APC_700_orSthLikeThat" // unused channel names & parts will be ignored but can be added for convenience
	},

	notes: "Whatever bla bla",
	flow_operator: "georg42",
	tx_operator: "charlesvdm",
	machine: "Fortessa II Weiss Lab",
	cell_line: "HEK293FT",
	transfection_reagent: "Lipo3K",
	rpm: "0",
	suspension: "False",
	transfection_plate: 24,
	grow_plate: "15cm",
	incubation_temp: 37.0,
	beads: "A04",
	transfection_protocol: "reverse_TX"
	// ... + anything you can think of that seems relevant to troubleshoot / distinguish between replicates
}

``` 

## Data files
#### Naming convention: `{sample name}.{xpname}.csv`. 
Example: `tube001.20220708-CVDM-uorfthings.csv`
Store in `experiments/{xpname}/data/` 

CSV file containing the actual MEFL intensity reads. One file per sample. 
One cell per row. Columns = fluo channel intensity in MEFL. Comma separated. First row = column names.


# SUMMARY: How to save, format, and share XP data:

Everything is stored on the MIT Dropbox /Biocomp folder. (That will be our root / for the rest of these instructions)
> !!USE THE NAMING CONVENTIONS DEFINED ABOVE!!

1. Write the recipe files that your XP implements (or check that they already exist). Save in /recipes/****.recipe.json5
2. Write the xp file and save in /experiments/{xpname}/{xpname}.xp.json5
3. Preprocess your data to make sure everything is in MEFL, and debris have been gated out. Each tube has its own individual file. Remove all the columns except the fluo channel intensities. If you want you can keep a copy of the raw unprocessed data under /experiments/{xpname}/raw_data/
4. Put each processed data file under /experiments/{xpname}/data/{samplename}.{xpname}.csv


That's it!


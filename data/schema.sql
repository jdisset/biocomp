CREATE TABLE IF NOT EXISTS part (
	name TEXT PRIMARY kEY NOT NULL,
	category TEXT NOT NULL,
	FOREIGN KEY (category) REFERENCES category(name)
);

CREATE TABLE IF NOT EXISTS category (
	name TEXT PRIMARY KEY NOT NULL,
	transcripted BOOLEAN NOT NULL,
	translated BOOLEAN NOT NULL
);

-- create with: cat schema.sql | sqlite3 database.db
-- import csv with: sqlite3 -csv database.db ".import --skip 1 part.csv part"

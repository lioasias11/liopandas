import liopandas as lp

df = lp.read_csv("cleaned_merged_seasons.csv")

#df = df.groupby(["position", "season_x"]).sum()[["position","season_x","total_points", "assists", "goals_scored", "clean_sheets", "red_cards", "yellow_cards"]]
#print(df.head(60))

df1 = lp.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [28, 34, 22]})
df2 = lp.DataFrame({"name": ["Alice", "Bob", "Charles"], "age": [28, 35, 22]})

print(df1.compare(df2))





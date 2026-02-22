import liopandas as lp

df = lp.read_csv("cleaned_merged_seasons.csv")

print(df)

df = df.groupby(["position", "season_x"]).sum()[["position","season_x","total_points", "assists", "goals_scored", "clean_sheets", "red_cards", "yellow_cards"]]
print(df)





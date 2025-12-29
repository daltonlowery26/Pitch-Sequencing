# %% package
import polars as pl
import os
os.chdir('C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/')

# %% load df
cols_to_remove = ['', 'spin_rate_deprecated', 'des', 'break_angle_deprecated', 'break_length_deprecated', 'spin_dir', 'hit_location',
    'game_type', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning_topbot', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'fielder_2',
    'fielder_2','fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'home_score', 'away_score', 
    'bat_score', 'fld_score', 'post_away_score', 'post_home_score','sv_id', 'post_bat_score', 'post_fld_score', 'home_score_diff', 'bat_score_diff',
    'age_pit_legacy', 'home_win_exp','woba_denom', 'if_fielding_alignment', 'babip_value', 'launch_speed_angle', 'iso_value', 'of_fielding_alignment', 
    'delta_home_win_exp','bat_win_exp', 'age_pit_legacy', 'age_bat_legacy','hyper_speed', 'age_pit', 'age_bat', 'pitcher_days_since_prev_game', 
    'batter_days_since_prev_game', 'pitcher_days_until_next_game', 'batter_days_until_next_game', 'delta_pitcher_run_exp']

unclean = (pl.scan_csv('raw_data/pitch_2015_2026.csv')
    .filter(pl.col("game_type") == 'R')
    .select(pl.exclude(cols_to_remove))
    .with_columns(pl.concat_str(pl.col('balls'), pl.col('strikes'), separator="-").alias("count"))
    .rename({'batter':'batter_id', 'pitcher':'pitcher_id', 
            'player_name':'pitcher_name', 'release_pos_z':'release_height', 
            'release_pos_x':'release_x', 'stand':'b_stand', 'estimated_woba_using_speedangle':'xwoba',
            'estimated_ba_using_speedangle':'xba'
            }))
# %% abs strike, batter name, pitcher postion (fangraphs merge)
bat_names = (
    pl.scan_csv('raw_data/batter_names.csv')
    .rename({'last_name, first_name':'batter_name'})
    .select(pl.exclude('year'))
    .unique()
)
# add batter names
unclean = unclean.join(bat_names, left_on='batter_id', right_on='player_id', how='left')
# abs ball strike
unclean = unclean.with_columns(
    (pl.col('plate_z').is_between(pl.col('sz_bot'), pl.col('sz_top')) 
    & pl.col('plate_x').is_between(-0.7083, 0.7083)).alias('abs_strike'))
# names at start
start = ['game_date', 'pitcher_name', 'pitcher_id','batter_name', 'batter_id']
unclean = unclean.select(
    pl.col(start),
    pl.exclude(start) # get rest of cols
).collect(engine="streaming")

# %% adding change in woba value, value of individ pitch
woba = pl.read_csv('raw_data/woba.csv')
woba = woba.unpivot(
    index="year",
    variable_name="count",
    value_name="woba_value"
)
woba = woba.rename({'woba_value':'count_value'})
unclean = unclean.join(woba, right_on=['year', 'count'], left_on=['game_year', 'count'], how="left")
unclean = unclean.sort(by=['game_pk', 'at_bat_number', 'pitch_number'])
unclean = unclean.with_columns(
    pl.col('count_value')
    .diff(n=-1)
    .over("game_pk", "at_bat_number")
    .alias("pitch_value")
    .fill_null(pl.col('count_value') - pl.col('woba_value'))
)

# %% swings
swing = ['hit_into_play', 'foul', 'swinging_strike', 'foul_tip', 'swinging_strike_blocked']
unclean = unclean.with_columns(pl.col('description').is_in(swing).alias('swing'))


# %% clean csv
unclean.write_csv("cleaned_data/pitch_2015_2026.csv")

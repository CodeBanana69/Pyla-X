"""Aimbot playstyles registered alongside the .pyla library."""

AIMBOT_PLAYSTYLES = (
    {
        "name": "Follower + Aimbot",
        "filename": "follower_aimbot.pyla",
        "base": "follower.pyla",
    },
    {
        "name": "Slarckvul's Aggressive + Aimbot",
        "filename": "slarckvul_aggressive_aimbot.pyla",
        "base": "universal_smart_v5_Slarckvul_Eddition.pyla",
    },
    {
        "name": "Showdown Survivor + Aimbot",
        "filename": "showdown_survivor_aimbot.pyla",
        "base": "showdown_survivor.pyla",
    },
)


def registered_aimbot_names():
    return [item["name"] for item in AIMBOT_PLAYSTYLES]

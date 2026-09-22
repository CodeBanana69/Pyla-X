import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bot_instances import AdbScrcpyService, BotInstance, InstanceRegistry, read_recent_matches
from discord_bot import DiscordBot


class FakeResponse:
    def __init__(self):
        self.messages = []
        self.kwargs = []

    async def send_message(self, content=None, **kwargs):
        self.messages.append(content)
        self.kwargs.append(kwargs)

    async def send(self, content=None, **kwargs):
        await self.send_message(content, **kwargs)


class FakeInteraction:
    def __init__(self, user_id=1):
        self.user = SimpleNamespace(id=user_id)
        self.response = FakeResponse()
        self.followup = self.response


class FakeRuntime:
    def __init__(self, name, state="running", running=True):
        self.name = name
        self.state = state
        self.running = running
        self.session_started_at = 1000.0 if running else None

    def get_status(self):
        return {
            "state": self.state,
            "is_running": self.running,
            "last_error": "",
            "session_started_at": self.session_started_at,
        }

    def start_current_queue(self, _discord_bot):
        self.running = True
        self.state = "running"
        return {"ok": True, "message": f"{self.name} started."}

    def stop(self):
        self.running = False
        self.state = "idle"
        return {"ok": True, "message": f"{self.name} stopped."}

    def pause(self):
        self.state = "pausing"
        return {"ok": True, "message": f"{self.name} pausing."}


class FakeAdbService:
    def __init__(self):
        self.calls = []

    def restart(self, instance):
        self.calls.append(instance.instance_id)
        return {
            "ok": True,
            "message": f"ADB and scrcpy restarted for {instance.instance_id}.",
            "adb_restarted": True,
            "scrcpy_restarted": True,
            "instance_id": instance.instance_id,
        }


def make_instance(instance_id, brawlers=None, matches=None, session=None, queue=None, adb_service=None, runtime=None):
    instance = BotInstance(
        instance_id=instance_id,
        runtime_manager=runtime or FakeRuntime(instance_id),
        data_service=None,
        adb_service=adb_service or FakeAdbService(),
        persist_player_tag=False,
    )
    instance._queue = [] if queue is None else queue
    instance.matches = [] if matches is None else matches
    instance.brawler_trophies = [] if brawlers is None else brawlers
    instance.update_session(session or {})
    return instance


def make_bot(instances, active=None):
    registry = InstanceRegistry()
    for instance in instances:
        registry.register(instance)
    if active:
        registry.activate(active)
    return DiscordBot(instances[0].runtime_manager, None, instance_registry=registry)


def run(coro):
    return asyncio.run(coro)


class DiscordCommandTests(unittest.TestCase):
    def test_commands_are_registered(self):
        bot = make_bot([make_instance("default")])
        names = {command.name for command in bot.tree.walk_commands()}
        self.assertTrue({
            "activate_instance",
            "push_all",
            "session",
            "switch_player_tag",
            "match_history",
            "restart_adb",
        }.issubset(names))

    def test_push_all_enqueues_only_brawlers_below_threshold_on_active_instance(self):
        phone = make_instance("phone", brawlers=[
            {"brawler": "Shelly", "trophies": 100, "win_streak": 2},
            {"brawler": "Colt", "trophies": 800, "win_streak": 0},
            {"brawler": "Bull", "trophies": 400, "win_streak": 1, "wins": 3},
        ], queue=[{"brawler": "Nita", "trophies": 900, "type": "trophies", "push_until": 1000, "wins": 0, "automatically_pick": False, "win_streak": 0}])
        tablet = make_instance("tablet", brawlers=[
            {"brawler": "Spike", "trophies": 10},
        ])
        bot = make_bot([phone, tablet], active="phone")
        interaction = FakeInteraction()

        run(bot.handle_push_all(interaction, 500))

        queued = {item["brawler"]: item for item in phone.get_queue()}
        self.assertEqual(set(queued), {"Shelly", "Bull", "Nita"})
        self.assertEqual(queued["Shelly"]["push_until"], 500)
        self.assertEqual(queued["Shelly"]["trophies"], 100)
        self.assertEqual(queued["Bull"]["wins"], 3)
        self.assertEqual(queued["Nita"]["push_until"], 1000)
        self.assertEqual(tablet.get_queue(), [])
        self.assertIn("`phone`", interaction.response.messages[0])
        self.assertNotIn("Colt", interaction.response.messages[0])
        self.assertNotIn("Spike", interaction.response.messages[0])

    def test_commands_follow_the_instance_selected_by_activate_instance(self):
        phone = make_instance("phone", brawlers=[{"brawler": "Shelly", "trophies": 50}])
        tablet = make_instance("tablet", brawlers=[{"brawler": "Spike", "trophies": 20}])
        bot = make_bot([phone, tablet], active="phone")

        run(bot.handle_activate_instance(FakeInteraction(), "tablet"))
        self.assertEqual(bot.active_instance().instance_id, "tablet")

        run(bot.handle_push_all(FakeInteraction(), 100))
        self.assertEqual(phone.get_queue(), [])
        self.assertEqual([item["brawler"] for item in tablet.get_queue()], ["Spike"])

        created = FakeInteraction()
        run(bot.handle_activate_instance(created, "emulator two"))
        self.assertEqual(bot.active_instance().instance_id, "emulator two")
        self.assertIn("Created a new instance", created.response.messages[0])
        self.assertEqual(bot.active_instance().get_queue(), [])

    def test_session_reports_live_stats_for_the_active_instance(self):
        phone = make_instance("phone", session={
            "trophies": 642,
            "wins": 4,
            "losses": 1,
            "current_brawler": "Shelly",
            "mode": "brawlBall",
        })
        phone.set_player_tag("#P2Y")
        tablet = make_instance("tablet", session={
            "trophies": 10,
            "wins": 0,
            "losses": 9,
            "current_brawler": "Spike",
            "mode": "soloShowdown",
        })
        bot = make_bot([phone, tablet], active="tablet")
        interaction = FakeInteraction()

        run(bot.handle_session(interaction))

        message = interaction.response.messages[0]
        self.assertIn("`tablet`", message)
        self.assertIn("Trophies: 10", message)
        self.assertIn("Wins: 0", message)
        self.assertIn("Losses: 9", message)
        self.assertIn("Brawler: Spike", message)
        self.assertIn("Mode: soloShowdown", message)
        self.assertNotIn("Shelly", message)

        run(bot.handle_activate_instance(FakeInteraction(), "phone"))
        phone_session = FakeInteraction()
        run(bot.handle_session(phone_session))
        phone_message = phone_session.response.messages[0]
        self.assertIn("Trophies: 642", phone_message)
        self.assertIn("Wins: 4", phone_message)
        self.assertIn("Losses: 1", phone_message)
        self.assertIn("Brawler: Shelly", phone_message)
        self.assertIn("Mode: brawlBall", phone_message)
        self.assertIn("Player tag: #P2Y", phone_message)

    def test_switch_player_tag_updates_only_the_active_instance(self):
        phone = make_instance("phone")
        tablet = make_instance("tablet")
        phone.set_player_tag("P2Y")
        tablet.set_player_tag("8L9Q")
        bot = make_bot([phone, tablet], active="phone")

        updated = FakeInteraction()
        run(bot.handle_switch_player_tag(updated, "#2PP"))
        self.assertEqual(phone.get_player_tag(), "2PP")
        self.assertEqual(tablet.get_player_tag(), "8L9Q")
        self.assertIn("`phone`", updated.response.messages[0])
        self.assertIn("#2PP", updated.response.messages[0])

        rejected = FakeInteraction()
        run(bot.handle_switch_player_tag(rejected, "not-a-tag"))
        self.assertEqual(phone.get_player_tag(), "2PP")
        self.assertIn("Player tag is incorrect", rejected.response.messages[0])

    def test_match_history_limit_and_instance_are_respected(self):
        phone_matches = [
            {"brawler": "Shelly", "date_time": "2026-09-01 10:00", "result": "victory", "trophy_delta": 8, "mode": "brawlBall"},
            {"brawler": "Colt", "date_time": "2026-09-01 11:00", "result": "defeat", "trophy_delta": -4, "mode": "gemGrab"},
            {"brawler": "Bull", "date_time": "2026-09-01 12:00", "result": "victory", "trophy_delta": 8, "mode": "brawlBall"},
        ]
        tablet_matches = [
            {"brawler": "Spike", "date_time": "2026-09-02 09:00", "result": "defeat", "trophy_delta": -6, "playstyle_name": "Showdown"},
        ]
        bot = make_bot([
            make_instance("phone", matches=phone_matches),
            make_instance("tablet", matches=tablet_matches),
        ], active="phone")
        interaction = FakeInteraction()

        run(bot.handle_match_history(interaction, 2))

        message = interaction.response.messages[0]
        self.assertIn("`phone`", message)
        self.assertIn("showing 2", message)
        self.assertIn("Bull", message)
        self.assertIn("Colt", message)
        self.assertNotIn("Shelly", message)
        self.assertIn("(+8)", message)
        self.assertIn("(-4)", message)
        self.assertNotIn("Spike", message)

        run(bot.handle_activate_instance(FakeInteraction(), "tablet"))
        tablet_history = FakeInteraction()
        run(bot.handle_match_history(tablet_history, 10))
        tablet_message = tablet_history.response.messages[0]
        self.assertIn("Spike", tablet_message)
        self.assertIn("[Showdown]", tablet_message)
        self.assertNotIn("Bull", tablet_message)

        invalid = FakeInteraction()
        run(bot.handle_match_history(invalid, 0))
        self.assertIn("at least 1", invalid.response.messages[0])

    def test_restart_adb_uses_the_injected_service_for_the_active_instance(self):
        phone_adb = FakeAdbService()
        tablet_adb = FakeAdbService()
        phone = make_instance("phone", adb_service=phone_adb)
        tablet = make_instance("tablet", adb_service=tablet_adb)
        bot = make_bot([phone, tablet], active="phone")

        phone_result = FakeInteraction()
        run(bot.handle_restart_adb(phone_result))
        self.assertEqual(phone_adb.calls, ["phone"])
        self.assertEqual(tablet_adb.calls, [])
        self.assertIn("Instance: `phone`", phone_result.response.messages[0])
        self.assertTrue(phone_result.response.messages[0].startswith("Success"))

        run(bot.handle_activate_instance(FakeInteraction(), "tablet"))
        run(bot.handle_restart_adb(FakeInteraction()))
        self.assertEqual(phone_adb.calls, ["phone"])
        self.assertEqual(tablet_adb.calls, ["tablet"])

    def test_restart_adb_service_stays_on_injected_callables(self):
        events = []

        def stop_scrcpy(instance):
            events.append(("stop", instance.instance_id))

        def restart_adb(instance):
            events.append(("adb", instance.instance_id))

        def reconnect_scrcpy(instance):
            events.append(("scrcpy", instance.instance_id))
            return True

        service = AdbScrcpyService(
            restart_adb=restart_adb,
            stop_scrcpy=stop_scrcpy,
            reconnect_scrcpy=reconnect_scrcpy,
        )
        instance = make_instance("phone", adb_service=service)
        with mock.patch.dict("sys.modules", {"window_controller": None}):
            result = service.restart(instance)

        self.assertEqual(events, [("stop", "phone"), ("adb", "phone"), ("scrcpy", "phone")])
        self.assertTrue(result["ok"])
        self.assertTrue(result["adb_restarted"])
        self.assertTrue(result["scrcpy_restarted"])

    def test_spawned_instance_restart_uses_the_bot_service(self):
        shared = FakeAdbService()
        bot = make_bot([make_instance("default")])
        bot.adb_service = shared
        run(bot.handle_activate_instance(FakeInteraction(), "phone"))
        run(bot.handle_restart_adb(FakeInteraction()))
        self.assertEqual(shared.calls, ["phone"])

    def test_unauthorized_user_cannot_change_instance_state(self):
        phone = make_instance("phone", brawlers=[{"brawler": "Shelly", "trophies": 10}])
        bot = make_bot([phone])
        bot.get_authorized_user_id = lambda: 42
        interaction = FakeInteraction(user_id=7)

        run(bot.handle_push_all(interaction, 100))

        self.assertEqual(phone.get_queue(), [])
        self.assertEqual(interaction.response.messages, ["You are not authorized to use this command."])
        self.assertTrue(interaction.response.kwargs[0]["ephemeral"])

    def test_default_instance_is_created_when_no_registry_is_given(self):
        bot = DiscordBot(FakeRuntime("default"), None)
        self.assertEqual(bot.active_instance().instance_id, "default")
        self.assertEqual(bot.primary_instance().instance_id, "default")
        session = FakeInteraction()
        run(bot.handle_session(session))
        self.assertIn("`default`", session.response.messages[0])

    def test_token_comes_from_the_environment_or_config(self):
        bot = DiscordBot(FakeRuntime("default"), None)
        with mock.patch.dict(os.environ, {"PYLA_DISCORD_BOT_TOKEN": "env-token", "DISCORD_BOT_TOKEN": "other"}, clear=False):
            self.assertEqual(bot.get_discord_bot_token(), "env-token")
        with mock.patch.dict(os.environ, {"PYLA_DISCORD_BOT_TOKEN": "", "DISCORD_BOT_TOKEN": "fallback-token"}, clear=False):
            self.assertEqual(bot.get_discord_bot_token(), "fallback-token")
        with mock.patch.dict(os.environ, {"PYLA_DISCORD_BOT_TOKEN": "", "DISCORD_BOT_TOKEN": ""}, clear=False):
            with mock.patch("discord_bot.load_toml_as_dict", return_value={"discord_bot_token": "cfg-token"}):
                self.assertEqual(bot.get_discord_bot_token(), "cfg-token")

    def test_recent_match_reader_applies_the_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "match_history.csv"
            csv_path.write_text(
                "date_time,brawler_name,result,current_trophies,trophy_delta,new_winstreak,playstyle_hash,playstyle_name,playstyle_gamemodes,playstyle_brawlers,pyla_version,power_level\n"
                "2026-09-01T10:00:00,Shelly,victory,100,8,1,,,brawlBall,,,\n"
                ",,,,,,,,,,,\n"
                "2026-09-01T12:00:00,Bull,defeat,200,-4,0,,,gemGrab,,,\n"
                "2026-09-01T11:00:00,Colt,victory,150,8,2,,,,,\n",
                encoding="utf-8",
            )
            matches = read_recent_matches(csv_path, limit=2)
        self.assertEqual([match["brawler"] for match in matches], ["Bull", "Colt"])
        self.assertEqual(matches[0]["mode"], "gemGrab")
        self.assertEqual(matches[0]["trophy_delta"], -4)

    def test_push_all_uses_the_active_instance_data_service(self):
        class FakeData:
            def __init__(self):
                self.queue = []

            def list_brawler_trophies(self, _player_tag=None):
                return [
                    {"brawler": "Shelly", "trophies": 100, "win_streak": 1},
                    {"brawler": "Colt", "trophies": 900},
                ]

            def get_queue_data(self):
                return [dict(item) for item in self.queue]

            def save_queue_data(self, items):
                self.queue = [dict(item) for item in items]
                return self.queue

            def _assert_queue_editable(self):
                return None

        data = FakeData()
        bot = DiscordBot(FakeRuntime("default", state="idle", running=False), data)
        interaction = FakeInteraction()

        run(bot.handle_push_all(interaction, 500))

        self.assertEqual([item["brawler"] for item in data.queue], ["Shelly"])
        self.assertEqual(data.queue[0]["push_until"], 500)
        self.assertIn("`default`", interaction.response.messages[0])

    def test_tree_callback_routes_push_all(self):
        phone = make_instance("phone", brawlers=[{"brawler": "Shelly", "trophies": 10}])
        bot = make_bot([phone], active="phone")
        command = bot.tree.get_command("push_all")
        interaction = FakeInteraction()
        run(command.callback(interaction, trophy_threshold=100))
        self.assertEqual([item["brawler"] for item in phone.get_queue()], ["Shelly"])
        self.assertIn("`phone`", interaction.response.messages[0])


if __name__ == "__main__":
    unittest.main()

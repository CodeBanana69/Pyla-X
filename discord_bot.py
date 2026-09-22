import asyncio
import os
import time
from io import BytesIO

from discord import app_commands
import discord
from PIL import Image
from bot_instances import (
    AdbScrcpyService,
    BotInstance,
    InstanceRegistry,
    normalize_instance_name,
)
from utils import load_toml_as_dict
TIMEOUT = 300
DEFAULT_INSTANCE_ID = "default"


class DiscordBot:
    def __init__(self, runtime_manager, data_service, instance_registry=None, adb_service=None):
        self.adb_service = adb_service
        self.registry = instance_registry if instance_registry is not None else InstanceRegistry()
        if len(self.registry) == 0:
            self.registry.register(BotInstance(
                instance_id=DEFAULT_INSTANCE_ID,
                runtime_manager=runtime_manager,
                data_service=data_service,
                adb_service=adb_service or AdbScrcpyService(),
                persist_player_tag=True,
            ))
            self._primary_id = DEFAULT_INSTANCE_ID
        else:
            self._primary_id = self.registry.get_active().instance_id
        self.started = False
        self.commands_synced = False

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        self.client = discord.Client(intents=intents)
        self.tree = app_commands.CommandTree(self.client)
        self.register_events()
        self.register_commands()

    def active_instance(self) -> BotInstance:
        return self.registry.get_active()

    def primary_instance(self) -> BotInstance:
        instance = self.registry.get(self._primary_id)
        if instance is None:
            return self.active_instance()
        return instance

    @property
    def runtime_manager(self):
        return self.active_instance().runtime_manager

    @property
    def data_service(self):
        return self.active_instance().data_service

    @property
    def window_controller(self):
        return self.active_instance().window_controller

    @window_controller.setter
    def window_controller(self, value):
        self.active_instance().window_controller = value

    def set_window_controller(self, window_controller):
        self.primary_instance().window_controller = window_controller

    def publish_session(self, stats, instance_id=None):
        if instance_id is None:
            instance = self.primary_instance()
        else:
            instance = self.registry.get(instance_id)
        if instance is None:
            return
        instance.update_session(stats)

    def _spawn_instance(self, instance_name: str) -> BotInstance:
        service = self.adb_service if self.adb_service is not None else AdbScrcpyService()
        instance = BotInstance(
            instance_id=instance_name,
            adb_service=service,
            persist_player_tag=False,
        )
        instance._queue = []
        instance.matches = []
        instance.brawler_trophies = []
        return self.registry.register(instance)

    def get_discord_bot_token(self) -> str:
        for key in ("PYLA_DISCORD_BOT_TOKEN", "DISCORD_BOT_TOKEN"):
            value = str(os.environ.get(key, "")).strip()
            if value:
                return value
        return str(load_toml_as_dict("cfg/webhook_config.toml", cache=False).get("discord_bot_token", "")).strip()

    @staticmethod
    def _extract_discord_id(value):
        digits = "".join(ch for ch in str(value or "").strip() if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    def get_authorized_user_id(self):
        config = load_toml_as_dict("cfg/webhook_config.toml", cache=False)
        return self._extract_discord_id(config.get("discord_id", ""))

    def get_configured_guild_id(self):
        guild_id = str(load_toml_as_dict("cfg/webhook_config.toml", cache=False).get("discord_guild_id", "")).strip()
        if not guild_id:
            return None

        try:
            return int(guild_id)
        except ValueError:
            print(f"Invalid discord_guild_id in cfg/webhook_config.toml: {guild_id}")
            return None

    def get_configured_guild(self):
        guild_id = self.get_configured_guild_id()
        if not guild_id:
            return None

        return discord.Object(id=guild_id)

    async def require_authorized_user(self, interaction: discord.Interaction) -> bool:
        authorized_id = self.get_authorized_user_id()
        if authorized_id is None:
            return True

        if interaction.user.id == authorized_id:
            return True

        await interaction.response.send_message(
            "You are not authorized to use this command.",
            ephemeral=True
        )
        return False

    async def sync_commands(self):
        guild = self.get_configured_guild()
        if guild:
            self.tree.copy_global_to(guild=guild)
            commands = await self.tree.sync(guild=guild)
            return len(commands), "guild"

        commands = await self.tree.sync()
        return len(commands), "global"

    def register_events(self):
        @self.client.event
        async def on_ready():
            print(f"Discord bot logged in as {self.client.user}")
            if self.commands_synced:
                return

            configured_guild_id = self.get_configured_guild_id()
            if configured_guild_id:
                guild = discord.Object(id=configured_guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                print(f"Discord slash commands synced to guild {configured_guild_id}.")
            else:
                await self.tree.sync()
                print("Discord slash commands synced globally.")
            self.commands_synced = True

    async def handle_activate_instance(self, interaction: discord.Interaction, instance_name: str):
        if not await self.require_authorized_user(interaction):
            return
        try:
            name = normalize_instance_name(instance_name)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return

        created = self.registry.get(name) is None
        if created:
            self._spawn_instance(name)
        self.registry.activate(name)
        others = ", ".join(item["id"] for item in self.registry.list_instances())
        message = f"Active instance is now `{name}`."
        if created:
            message += " Created a new instance."
        message += f"\nInstances: {others}"
        await interaction.response.send_message(message, ephemeral=True)

    async def handle_push_all(self, interaction: discord.Interaction, trophy_threshold: int):
        if not await self.require_authorized_user(interaction):
            return
        instance = self.active_instance()
        try:
            result = instance.push_all(trophy_threshold)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return

        count = int(result.get("added_count") or 0)
        threshold = int(trophy_threshold)
        if result.get("reason") == "no_trophy_data":
            message = (
                f"No trophy data is available for `{instance.instance_id}`, "
                f"so nothing was queued under {threshold}."
            )
        elif count == 0:
            message = f"No brawlers under {threshold} trophies were found for `{instance.instance_id}`."
        else:
            brawlers = list(result.get("brawlers") or [])
            preview = brawlers[:30]
            names = ", ".join(preview)
            if len(brawlers) > len(preview):
                names += f", and {len(brawlers) - len(preview)} more"
            message = f"Queued {count} brawler(s) under {threshold} trophies on `{instance.instance_id}`."
            if names:
                message += f"\n{names}"
        await interaction.response.send_message(message, ephemeral=True)

    async def handle_session(self, interaction: discord.Interaction):
        if not await self.require_authorized_user(interaction):
            return
        snapshot = self.active_instance().get_session()
        trophies = snapshot.get("trophies")
        trophies_text = "unknown" if trophies is None else str(trophies)
        brawler = snapshot.get("current_brawler") or "none"
        mode = snapshot.get("mode") or "unknown"
        lines = [
            f"Session for `{snapshot['instance_id']}` ({snapshot.get('state') or 'idle'})",
            f"Trophies: {trophies_text}",
            f"Wins: {snapshot.get('wins', 0)}",
            f"Losses: {snapshot.get('losses', 0)}",
            f"Brawler: {brawler}",
            f"Mode: {mode}",
        ]
        if snapshot.get("player_tag"):
            lines.append(f"Player tag: #{snapshot['player_tag']}")
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    async def handle_switch_player_tag(self, interaction: discord.Interaction, player_tag: str):
        if not await self.require_authorized_user(interaction):
            return
        instance = self.active_instance()
        try:
            tag = instance.set_player_tag(player_tag)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.send_message(
            f"Player tag for `{instance.instance_id}` is now `#{tag}`.",
            ephemeral=True,
        )

    async def handle_match_history(self, interaction: discord.Interaction, limit: int = 10):
        if not await self.require_authorized_user(interaction):
            return
        instance = self.active_instance()
        try:
            matches = instance.get_match_history(limit)
        except ValueError as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return

        if not matches:
            await interaction.response.send_message(
                f"No recent matches for `{instance.instance_id}`.",
                ephemeral=True,
            )
            return

        lines = [f"Recent matches for `{instance.instance_id}` (showing {len(matches)}):"]
        for match in matches:
            brawler = match.get("brawler") or "Unknown"
            result = match.get("result") or "unknown"
            when = match.get("date_time") or "unknown time"
            delta = match.get("trophy_delta")
            try:
                delta_text = f" ({int(delta):+d})" if delta is not None and str(delta) != "" else ""
            except (TypeError, ValueError):
                delta_text = ""
            mode = match.get("mode") or match.get("playstyle_name") or ""
            mode_text = f" [{mode}]" if mode else ""
            lines.append(f"- {when} {brawler}: {result}{delta_text}{mode_text}")
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    async def handle_restart_adb(self, interaction: discord.Interaction):
        if not await self.require_authorized_user(interaction):
            return
        instance = self.active_instance()
        try:
            result = instance.restart_adb()
        except Exception as exc:
            await interaction.response.send_message(
                f"Failed to restart ADB for `{instance.instance_id}`: {exc}",
                ephemeral=True,
            )
            return
        prefix = "Success" if result.get("ok") else "Failed"
        await interaction.response.send_message(
            f"{prefix}! {result.get('message', '')}\nInstance: `{instance.instance_id}`",
            ephemeral=True,
        )

    def register_commands(self):
        @self.tree.command(
            name="screenshot",
            description="Get a screenshot of the current game window",
        )
        async def screenshot(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            if not self.window_controller:
                await interaction.response.send_message(
                    "Failed to take a screenshot, is the bot running ?",
                    ephemeral=True
                )
                return
            screenshot_frame = self.window_controller.screenshot()
            if screenshot_frame is None:
                await interaction.response.send_message(
                    "Failed to take a screenshot, is the bot running ?",
                    ephemeral=True
                )
                return

            screenshot_buffer = BytesIO()
            Image.fromarray(screenshot_frame).save(screenshot_buffer, format="PNG")
            screenshot_buffer.seek(0)

            await interaction.response.send_message(
                "Here's a screenshot of the current game window:",
                files=[discord.File(screenshot_buffer, filename="screenshot.png")],
                ephemeral=True
            )

        @self.tree.command(
            name="stop",
            description="Stops the bot once it finishes its current task",
        )
        async def stop(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            status = self.runtime_manager.get_status()
            if status.get("state") == "idle" or not status.get("is_running"):
                await interaction.response.send_message(
                    "The bot is not currently running.",
                    ephemeral=True
                )
                return
            elif status.get("state") == "stopping":
                await interaction.response.send_message(
                    "The bot is already stopping, please wait.",
                    ephemeral=True
                )
                return
            elif status.get("state") == "error":
                await interaction.response.send_message(
                    f"The bot is in an error state :\n{status['last_error']}\nPlease wait a few seconds or check the logs.",
                    ephemeral=True
                )
                return
            elif status.get("state") == "pausing":
                await interaction.response.send_message(
                    "The bot is currently pausing, please wait before trying to stop it.",
                    ephemeral=True
                )
                return
            else:
                stop = self.runtime_manager.stop()
                await interaction.response.send_message(
                    f"{('Success' if stop.get('ok') else 'Failed')} ! {stop.get('message', '')}",
                    ephemeral=True
                )

                start_time = time.time()
                while time.time() - start_time < TIMEOUT:
                    status = self.runtime_manager.get_status()
                    if status.get("state") == "idle":
                        user_id = self.get_authorized_user_id()
                        await interaction.followup.send(
                            f"{('' if user_id is None else f'<@{user_id}> ')}The bot has stopped.",
                            ephemeral=True
                        )
                        return
                    elif status.get("state") != "stopping":
                        user_id = self.get_authorized_user_id()
                        await interaction.followup.send(
                            f"{('' if user_id is None else f'<@{user_id}> ')}The bot has unexpectedly stopped stopping. Current state: {status.get('state')}.",
                            ephemeral=True
                        )
                        return
                    await asyncio.sleep(1)

        @self.tree.command(
            name="pause",
            description="Makes the bot pause once it reaches the lobby",
        )
        async def pause(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            status = self.runtime_manager.get_status()
            if status.get("state") == "idle" or not status.get("is_running"):
                await interaction.response.send_message(
                    "The bot is not currently running.",
                    ephemeral=True
                )
                return
            elif status.get("state") == "pausing":
                await interaction.response.send_message(
                    "The bot is already pausing, please wait.",
                    ephemeral=True
                )
                return
            elif status.get("state") == "paused":
                await interaction.response.send_message(
                    "The bot is already paused.",
                    ephemeral=True
                )
                return
            elif status.get("state") == "error":
                await interaction.response.send_message(
                    f"The bot is in an error state :\n{status['last_error']}\nPlease wait a few seconds or check the logs.",
                    ephemeral=True
                )
                return
            elif status.get("state") == "stopping":
                await interaction.response.send_message(
                    "The bot is currently stopping, pausing isn't available.",
                    ephemeral=True
                )
                return
            else:
                pause = self.runtime_manager.pause()
                await interaction.response.send_message(
                    f"{('Success' if pause.get('ok') else 'Failed')} ! {pause.get('message', '')}",
                    ephemeral=True
                )

                start_time = time.time()
                while time.time() - start_time < TIMEOUT:
                    status = self.runtime_manager.get_status()
                    if status.get("state") == "paused":
                        user_id = self.get_authorized_user_id()
                        await interaction.followup.send(
                            f"{('' if user_id is None else f'<@{user_id}> ')}The bot is now paused.",
                            ephemeral=True
                        )
                        return
                    elif status.get("state") != "pausing":
                        user_id = self.get_authorized_user_id()
                        await interaction.followup.send(
                            f"{('' if user_id is None else f'<@{user_id}> ')}The bot has unexpectedly stopped pausing. Current state: {status.get('state')}.",
                            ephemeral=True
                        )
                        return
                    await asyncio.sleep(1)


        @self.tree.command(
            name="start",
            description="Starts the bot if it's not already running",
        )
        async def start(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            start_result = self.runtime_manager.start_current_queue(self)
            await interaction.response.send_message(
                f"{('Success' if start_result.get('ok') else 'Failed')} ! {start_result.get('message', '')}",
                ephemeral=True
            )

        @self.tree.command(
            name="status",
            description="Returns the current status of the bot",
        )
        async def status(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            status = self.runtime_manager.get_status()
            if not status.get("is_running"):
                await interaction.response.send_message(
                    "The bot is currently not running.",
                    ephemeral=True
                )
                return

            state = status.get("state", "unknown").capitalize()
            last_error = status.get("last_error")
            message = f"Instance `{self.active_instance().instance_id}` is currently **{state}**."
            if last_error:
                message += f"\nLast error: {last_error}"

            active_playstyle = None
            if self.data_service is not None and hasattr(self.data_service, "get_playstyles_payload"):
                active_playstyle = (self.data_service.get_playstyles_payload() or {}).get("current")
            playstyle_name = active_playstyle.get("name") if active_playstyle else None
            message += f"\n Playstyle : {playstyle_name or 'None'}"
            message += "\n Queue : do `/view_queue` to see the current queue."
            await interaction.response.send_message(
                message,
                ephemeral=True
            )

        @self.tree.command(
            name="restart_brawl_stars",
            description="Restarts Brawl Stars if the bot is running",
        )
        async def restart_brawl_stars(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            status = self.runtime_manager.get_status()
            if status.get("state") == "idle" or not status.get("is_running"):
                await interaction.response.send_message(
                    "The bot is not currently running.",
                    ephemeral=True
                )
                return
            if not self.window_controller:
                await interaction.response.send_message(
                    "This instance has no game window attached.",
                    ephemeral=True
                )
                return
            await interaction.response.send_message(
                f"Restarting brawl stars !",
                ephemeral=True
            )
            self.window_controller.restart_brawl_stars()

        @self.tree.command(
            name="view_queue",
            description="View the current queue of the bot",
        )
        async def view_queue(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            queue = self.active_instance().get_queue()
            if not queue:
                await interaction.response.send_message(
                    "The queue is currently empty.",
                    ephemeral=True
                )
                return

            message = f"Current queue for `{self.active_instance().instance_id}`:\n"
            responded = False

            for queue_item in queue:
                brawler = queue_item.get("brawler", "Unknown")
                push_type = queue_item.get("type", "Unknown")
                target_amount = queue_item.get("push_until", "Unknown")
                current_amount = queue_item.get("trophies") if push_type == "trophies" else queue_item.get("wins")
                auto_pick = queue_item.get("automatically_pick", False)
                message += f"- {brawler} : {current_amount}/{target_amount} {push_type} {('(Automatically picked)' if auto_pick else '')}\n"

                if len(message) > 1500:
                    if not responded:
                        await interaction.response.send_message(message, ephemeral=True)
                        responded = True
                    else:
                        await interaction.followup.send(message, ephemeral=True)
                    message = ""

            if message:
                if not responded:
                    await interaction.response.send_message(message, ephemeral=True)
                else:
                    await interaction.followup.send(message, ephemeral=True)

        @self.tree.command(
            name="help",
            description="Show the list of available commands",
        )
        async def help_command(interaction: discord.Interaction):
            if not await self.require_authorized_user(interaction):
                return

            commands = {
                "screenshot": "Get a screenshot of the current game window (only works when the bot is running)",
                "stop": "Makes the bot stop once it reaches the lobby",
                "pause": "Makes the bot pause once it reaches the lobby",
                "start": "Starts the bot if it's not already running",
                "status": "Returns the current status of the bot",
                "restart_brawl_stars": "Restarts Brawl Stars if the bot is running",
                "view_queue": "View the current queue of the bot",
                "add_to_queue": "**Premium Only:** Add a brawler to the queue remotely.",
                "remove_from_queue": "**Premium Only:** Remove a brawler from the queue remotely.",
                "clear_queue": "**Premium Only:** Clear the current queue remotely.",
                "activate_instance": "Select which bot instance the other commands control.",
                "push_all": "Queue every brawler under a trophy threshold on the active instance.",
                "session": "Show live session stats for the active instance.",
                "switch_player_tag": "Change the player tag on the active instance.",
                "match_history": "Show recent matches for the active instance.",
                "restart_adb": "Restart ADB and scrcpy for the active instance.",
                "activate_playstyle": "**Premium Only:** Activate a playstyle remotely.",
            }
            message = "**Available commands:**\n" + "\n".join(f"- `{command}`: {description}" for command, description in commands.items())
            message += "\n\nCommands apply to the instance selected with `/activate_instance`."
            message += "\n\n**Unlock Premium:** Visit https://pyla-ai.angelfirela.dev/premium for additional features and commands."
            await interaction.response.send_message(
                message,
                ephemeral=True
            )

        @self.tree.command(
            name="activate_instance",
            description="Select the bot instance that remote commands control",
        )
        @app_commands.describe(instance_name="Instance name. A new instance is created if it does not exist.")
        async def activate_instance(interaction: discord.Interaction, instance_name: str):
            await self.handle_activate_instance(interaction, instance_name)

        @self.tree.command(
            name="push_all",
            description="Queue every brawler below a trophy threshold",
        )
        @app_commands.describe(trophy_threshold="Brawlers with fewer trophies than this are queued")
        async def push_all(interaction: discord.Interaction, trophy_threshold: int):
            await self.handle_push_all(interaction, trophy_threshold)

        @self.tree.command(
            name="session",
            description="Show live session stats for the active instance",
        )
        async def session(interaction: discord.Interaction):
            await self.handle_session(interaction)

        @self.tree.command(
            name="switch_player_tag",
            description="Change the player tag on the active instance",
        )
        @app_commands.describe(player_tag="Brawl Stars player tag")
        async def switch_player_tag(interaction: discord.Interaction, player_tag: str):
            await self.handle_switch_player_tag(interaction, player_tag)

        @self.tree.command(
            name="match_history",
            description="Show recent matches for the active instance",
        )
        @app_commands.describe(limit="How many recent matches to show")
        async def match_history(interaction: discord.Interaction, limit: int = 10):
            await self.handle_match_history(interaction, limit)

        @self.tree.command(
            name="restart_adb",
            description="Restart ADB and scrcpy for the active instance",
        )
        async def restart_adb(interaction: discord.Interaction):
            await self.handle_restart_adb(interaction)

    def run_bot(self):
        discord_bot_token = self.get_discord_bot_token()
        if not discord_bot_token:
            print("Discord bot token is not configured. Skipping Discord bot startup.")
            return
        if self.started:
            return

        self.started = True
        try:
            self.client.run(discord_bot_token)
        finally:
            self.started = False

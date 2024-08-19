from keras.preprocessing.text import Tokenizer

# List of all card names
card_names = [
    "Bash", "Defend", "Strike", "Anger", "Armaments", "Body Slam", "Clash", "Cleave", 
    "Clothesline", "Flex", "Havoc", "Headbutt", "Heavy Blade", "Iron Wave", "Perfected Strike", 
    "Pommel Strike", "Shrug It Off", "Sword Boomerang", "Thunderclap", "True Grit", "Twin Strike", 
    "Warcry", "Wild Strike", "Battle Trance", "Blood for Blood", "Bloodletting", "Burning Pact", 
    "Carnage", "Combust", "Dark Embrace", "Disarm", "Dropkick", "Dual Wield", "Entrench", "Evolve", 
    "Feel No Pain", "Fire Breathing", "Flame Barrier", "Ghostly Armor", "Hemokinesis", "Infernal Blade", 
    "Inflame", "Intimidate", "Metallicize", "Power Through", "Pummel", "Rage", "Rampage", "Reckless Charge", 
    "Rupture", "Searing Blow", "Second Wind", "Seeing Red", "Sentinel", "Sever Soul", "Shockwave", 
    "Spot Weakness", "Uppercut", "Whirlwind", "Barricade", "Berserk", "Bludgeon", "Brutality", "Corruption", 
    "Demon Form", "Double Tap", "Exhume", "Feed", "Fiend Fire", "Immolate", "Impervious", "Juggernaut", 
    "Limit Break", "Offering", "Reaper", "Defend", "Neutralize", "Strike", "Survivor", "Acrobatics", 
    "Backflip", "Bane", "Blade Dance", "Cloak and Dagger", "Dagger Spray", "Dagger Throw", "Deadly Poison", 
    "Deflect", "Dodge and Roll", "Flying Knee", "Outmaneuver", "Piercing Wail", "Poisoned Stab", "Prepared", 
    "Quick Slash", "Slice", "Sneaky Strike", "Sucker Punch", "Accuracy", "All-Out Attack", "Backstab", 
    "Blur", "Bouncing Flask", "Calculated Gamble", "Caltrops", "Catalyst", "Choke", "Concentrate", 
    "Crippling Cloud", "Dash", "Distraction", "Endless Agony", "Escape Plan", "Eviscerate", "Expertise", 
    "Finisher", "Flechettes", "Footwork", "Heel Hook", "Infinite Blades", "Leg Sweep", "Masterful Stab", 
    "Noxious Fumes", "Predator", "Reflex", "Riddle with Holes", "Setup", "Skewer", "Tactician", 
    "Terror", "Well-Laid Plans", "A Thousand Cuts", "Adrenaline", "After Image", "Alchemize", 
    "Bullet Time", "Burst", "Corpse Explosion", "Die Die Die", "Doppelganger", "Envenom", 
    "Glass Knife", "Grand Finale", "Malaise", "Nightmare", "Phantasmal Killer", "Storm of Steel", 
    "Tools of the Trade", "Unload", "Wraith Form", "Defend", "Dualcast", "Strike", "Zap", 
    "Ball Lightning", "Barrage", "Beam Cell", "Charge Battery", "Claw", "Cold Snap", "Compile Driver", 
    "Coolheaded", "Go for the Eyes", "Hologram", "Leap", "Rebound", "Recursion", "Stack", 
    "Steam Barrier", "Streamline", "Sweeping Beam", "TURBO", "Aggregate", "Auto-Shields", 
    "Blizzard", "Boot Sequence", "Bullseye", "Capacitor", "Chaos", "Chill", "Consume", 
    "Darkness", "Defragment", "Doom and Gloom", "Double Energy", "Equilibrium", "FTL", 
    "Force Field", "Fusion", "Genetic Algorithm", "Glacier", "Heatsinks", "Hello World", 
    "Loop", "Melter", "Overclock", "Recycle", "Reinforced Body", "Reprogram", "Rip and Tear", 
    "Scrape", "Self Repair", "Skim", "Static Discharge", "Storm", "Sunder", "Tempest", 
    "White Noise", "All for One", "Amplify", "Biased Cognition", "Buffer", "Core Surge", 
    "Creative AI", "Echo Form", "Electrodynamics", "Fission", "Hyperbeam", "Machine Learning", 
    "Meteor Strike", "Multi-Cast", "Rainbow", "Reboot", "Seek", "Thunder Strike", 
    "Eruption", "Strike", "Vigilance", "Bowling Bash", "Consecrate", "Crescendo", 
    "Crush Joints", "Cut Through Fate", "Empty Body", "Empty Fist", "Evaluate", 
    "Flurry of Blows", "Flying Sleeves", "Follow-Up", "Halt", "Just Lucky", "Pressure Points", 
    "Prostrate", "Protect", "Sash Whip", "Third Eye", "Tranquility", "Battle Hymn", 
    "Carve Reality", "Collect", "Conclude", "Deceive Reality", "Empty Mind", "Fasting", 
    "Fear No Evil", "Foreign Influence", "Foresight", "Indignation", "Inner Peace", 
    "Like Water", "Meditate", "Mental Fortress", "Nirvana", "Perseverance", "Pray", 
    "Reach Heaven", "Rushdown", "Sanctity", "Sands of Time", "Signature Move", "Simmering Fury", 
    "Study", "Swivel", "Talk to the Hand", "Tantrum", "Wallop", "Wave of the Hand", "Weave", 
    "Wheel Kick", "Windmill Strike", "Worship", "Wreath of Flame", "Alpha", "Blasphemy", 
    "Brilliance", "Conjure Blade", "Deus Ex Machina", "Deva Form", "Devotion", "Establishment", 
    "Judgment", "Lesson Learned", "Master Reality", "Omniscience", "Ragnarok", "Scrawl", 
    "Spirit Shield", "Vault", "Wish", "Bandage Up", "Blind", "Dark Shackles", "Deep Breath", 
    "Discovery", "Dramatic Entrance", "Enlightenment", "Finesse", "Flash of Steel", "Forethought", 
    "Good Instincts", "Impatience", "Jack of All Trades", "Madness", "Mind Blast", "Panacea", 
    "Panic Button", "Purity", "Swift Strike", "Trip", "Apotheosis", "Chrysalis", 
    "Hand of Greed", "Magnetism", "Master of Strategy", "Mayhem", "Metamorphosis", 
    "Panache", "Sadistic Nature", "Secret Technique", "Secret Weapon", "The Bomb", 
    "Thinking Ahead", "Transmutation", "Violence", "Apparition", "Become Almighty", 
    "Beta", "Bite", "Expunger", "Fame and Fortune", "Insight", "J.A.X.", "Live Forever", 
    "Miracle", "Omega", "Ritual Dagger", "Safety", "Shiv", "Smite", "Through Violence", 
    "Clumsy", "Decay", "Doubt", "Injury", "Normality", "Pain", "Parasite", "Regret", 
    "Shame", "Writhe", "Ascender's Bane", "Curse of the Bell", "Necronomicurse", 
    "Pride", "Burn", "Dazed", "Slimed", "Void", "Wound"
]

# Initialize the tokenizer
card_tokenizer = Tokenizer()
card_tokenizer.fit_on_texts(card_names)

card_types = ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE"]

# Initialize the tokenizer
card_type_tokenizer = Tokenizer()
card_type_tokenizer.fit_on_texts(card_types)

# Define the card rarities
card_rarities = ["BASIC", "SPECIAL", "COMMON", "UNCOMMON", "RARE", "CURSE"]

# Initialize the tokenizer
card_rarity_tokenizer = Tokenizer()
card_rarity_tokenizer.fit_on_texts(card_rarities)

intents = [
    "ATTACK", "ATTACK_BUFF", "ATTACK_DEBUFF", "ATTACK_DEFEND", "BUFF", "DEBUFF",
    "STRONG_DEBUFF", "DEBUG", "DEFEND", "DEFEND_DEBUFF", "DEFEND_BUFF", "ESCAPE",
    "MAGIC", "NONE", "SLEEP", "STUN", "UNKNOWN"
]


# Initialize the intent tokenizer
intent_tokenizer = Tokenizer()
intent_tokenizer.fit_on_texts(intents)

# List of all monster IDs
monster_ids = [
    "AcidSlime_L", "AcidSlime_M", "AcidSlime_S", "ApologySlime", "Cultist", "FungiBeast", 
    "GremlinFat", "GremlinNob", "GremlinThief", "GremlinTsundere", "GremlinWarrior", 
    "GremlinWizard", "Hexaghost", "HexaghostBody", "HexaghostOrb", "JawWorm", "Lagavulin", 
    "Looter", "LouseDefensive", "LouseNormal", "Sentry", "SlaverBlue", "SlaverRed", "SlimeBoss", 
    "SpikeSlime_L", "SpikeSlime_M", "SpikeSlime_S", "TheGuardian", "BanditBear", "BanditLeader", 
    "BanditPointy", "BookOfStabbing", "BronzeAutomaton", "BronzeOrb", "Byrd", "Centurion", "Champ", 
    "Chosen", "GremlinLeader", "Healer", "Mugger", "ShelledParasite", "SnakePlant", "Snecko", 
    "SphericGuardian", "TaskMaster", "TheCommector", "TorchHead", "AwakenedOne", "Darkling", "Deca", 
    "Donu", "Exploder", "GiantHead", "Maw", "Nemesis", "OrbWalker", "Reptomancer", "Repulsor", 
    "SnakeDagger", "Spiker", "SpireGrowth", "TimeEater", "Transient", "WritingMass, FuzzyLouseDefensive, FuzzyLouseNormal"
]

# Initialize the monster ID tokenizer
monster_id_tokenizer = Tokenizer()
monster_id_tokenizer.fit_on_texts(monster_ids)


screen_types = [
    "EVENT", "CHEST", "SHOP_ROOM", "REST", "CARD_REWARD", 
    "COMBAT_REWARD", "MAP", "BOSS_REWARD", "SHOP_SCREEN", 
    "GRID", "HAND_SELECT", "GAME_OVER", "COMPLETE", "NONE"
]

# Initialize the screen type tokenizer
screen_type_tokenizer = Tokenizer()
screen_type_tokenizer.fit_on_texts(screen_types)

powers = [
    "Accuracy", "After Image", "Amplify", "Anger", "Angry", "Artifact", "Attack Burn", "Barricade", 
    "BackAttack", "BeatOfDeath", "Bias", "Berserk", "Blur", "Brutality", "Buffer", "Burst", 
    "Choked", "Collect", "Combust", "Confusion", "Converse", "Constricted", "CorpseExplosionPower", 
    "Corruption", "Creative AI", "Curiosity", "Curl Up", "Dark Embrace", "Demon Form", "Dexterity", 
    "Double Damage", "Double Tap", "Draw Card", "Draw", "Draw Reduction", "DuplicationPower", 
    "Echo Form", "Electro", "EnergizedBlue", "Enrage", "Energized", "Entangled", "Envenom", "Equilibrium", 
    "Evolve", "Explosive", "Fading", "Feel No Pain", "Fire Breathing", "Flame Barrier", "Flight", 
    "Focus", "Nullify Attack", "Frail", "Shackled", "Generic Strength Up Power", "GrowthPower", 
    "Heatsink", "Hello", "Infinite Blades", "Hex", "IntangiblePlayer", "Intangible", "Invincible", 
    "Juggernaut", "Lightning Mastery", "Lockon", "Loop", "DexLoss", "Flex", "Magnetism", 
    "Malleable", "Mayhem", "Metallicize", "Minion", "Mode Shift", "Next Turn Block", "Night Terror", 
    "NoBlockPower", "No Draw", "Poison", "Painful Stabs", "Panache", "Pen Nib", "Phantasmal", 
    "Plated Armor", "Rage", "Compulsive", "Rebound", "RechargingCore", "Regenerate", "Repair", 
    "Life Link", "Retain Cards", "Ritual", "Rupture", "Sadistic", "Sharp Hide", "Shifting", 
    "Skill Burn", "Slow", "Split", "Spore Cloud", "Stasis", "StaticDischarge", "Storm", 
    "Strength", "StrikeUp", "Surrounded", "Thievery", "Thorns", "Thousand Cuts", "Time Warp", 
    "Tools Of The Trade", "TheBomb", "Unawakened", "Vulnerable", "Weakened", "Winter", "Wraith Form v2"
]


power_tokenizer = Tokenizer()
power_tokenizer.fit_on_texts(powers)

# List of all possible map symbols
map_symbols = ["?", "$", "T", "M", "E", "R"]

# Initialize the tokenizer for map symbols
map_symbol_tokenizer = Tokenizer()
map_symbol_tokenizer.fit_on_texts(map_symbols)


from keras.preprocessing.text import Tokenizer

# Complete list of all possible relics
relics_list = [
    "Burning Blood", "Cracked Core", "Pure Water", "Ring of the Snake", "Akabeko", "Anchor",
    "Ancient Tea Set", "Art of War", "Bag of Marbles", "Bag of Preparation", "Blood Vial", 
    "Bronze Scales", "Centennial Puzzle", "Ceramic Fish", "Damaru", "Data Disk", "Dream Catcher", 
    "Happy Flower", "Juzu Bracelet", "Lantern", "Maw Bank", "Meal Ticket", "Nunchaku", 
    "Oddly Smooth Stone", "Omamori", "Orichalcum", "Pen Nib", "Potion Belt", "Preserved Insect", 
    "Red Skull", "Regal Pillow", "Smiling Mask", "Snecko Skull", "Strawberry", "The Boot", 
    "Tiny Chest", "Toy Ornithopter", "Vajra", "War Paint", "Whetstone", "Blue Candle", 
    "Bottled Flame", "Bottled Lightning", "Bottled Tornado", "Darkstone Periapt", "Duality", 
    "Eternal Feather", "Frozen Egg", "Gold-Plated Cables", "Gremlin Horn", "Horn Cleat", 
    "Ink Bottle", "Kunai", "Letter Opener", "Matryoshka", "Meat on the Bone", "Mercury Hourglass", 
    "Molten Egg", "Mummified Hand", "Ninja Scroll", "Ornamental Fan", "Pantograph", "Paper Krane", 
    "Paper Phrog", "Pear", "Question Card", "Self-Forming Clay", "Shuriken", "Singing Bowl", 
    "Strike Dummy", "Sundial", "Symbiotic Virus", "Teardrop Locket", "The Courier", "Toxic Egg", 
    "White Beast Statue", "Bird-Faced Urn", "Calipers", "Captain's Wheel", "Champion Belt", 
    "Charon's Ashes", "Cloak Clasp", "Dead Branch", "Du-Vu Doll", "Emotion Chip", "Fossilized Helix", 
    "Gambling Chip", "Ginger", "Girya", "Golden Eye", "Ice Cream", "Incense Burner", "Lizard Tail", 
    "Magic Flower", "Mango", "Old Coin", "Peace Pipe", "Pocketwatch", "Prayer Wheel", "Shovel", 
    "Stone Calendar", "The Specimen", "Thread and Needle", "Tingsha", "Torii", "Tough Bandages", 
    "Tungsten Rod", "Turnip", "Unceasing Top", "Wing Boots", "Astrolabe", "Black Blood", "Black Star", 
    "Busted Crown", "Calling Bell", "Coffee Dripper", "Cursed Key", "Ectoplasm", "Empty Cage", 
    "Frozen Core", "Fusion Hammer", "Holy Water", "Hovering Kite", "Inserter", "Mark of Pain", 
    "Nuclear Battery", "Pandora's Box", "Philosopher's Stone", "Ring of the Serpent", "Runic Cube", 
    "Runic Dome", "Runic Pyramid", "Sacred Bark", "Slaver's Collar", "Snecko Eye", "Sozu", 
    "Tiny House", "Velvet Choker", "Violet Lotus", "Wrist Blade", "Brimstone", "Cauldron", 
    "Chemical X", "Clockwork Souvenir", "Dolly's Mirror", "Frozen Eye", "Hand Drill", 
    "Lee's Waffle", "Medical Kit", "Melange", "Membership Card", "Orange Pellets", "Orrery", 
    "Prismatic Shard", "Runic Capacitor", "Sling of Courage", "Strange Spoon", "The Abacus", 
    "Toolbox", "Twisted Funnel", "Bloody Idol", "Cultist Headpiece", "Enchiridion", "Face Of Cleric", 
    "Golden Idol", "Gremlin Visage", "Mark of the Bloom", "Mutagenic Strength", "N'loth's Gift", 
    "N'loth's Hungry Face", "Necronomicon", "Neow's Lament", "Nilry's Codex", "Odd Mushroom", 
    "Red Mask", "Spirit Poop", "Ssserpent Head", "Warped Tongs"
]

# Initialize the tokenizer for relics
relic_tokenizer = Tokenizer()
relic_tokenizer.fit_on_texts(relics_list)

# List of all potion types
potion_types = [
    "Ambrosia", "Ancient Potion", "AttackPotion", "BlessingOfTheForge", "Block Potion", "BottledMiracle",
    "BloodPotion", "ColorlessPotion", "CultistPotion", "CunningPotion", "Dexterity Potion", "DistilledChaos",
    "DuplicationPotion", "ElixirPotion", "Energy Potion", "EntriopicBrew", "EssenceOfDarkness", "EssenceOfSteel",
    "Explosive Potion", "FairyPotion", "FearPotion", "Fire Potion", "FocusPotion", "Fruit Juice", "GamblersBrew",
    "GhostInAJar", "HeartOfIron", "LiquidBronze", "LiquidMemories", "Poison Potion", "PotionOfCapacity", "PotionSlot",
    "PowerPotion", "Regen Potion", "SkillPotion", "SmokeBomb", "SneckoOil", "SpeedPotion", "StancePotion",
    "SteroidPotion", "Strength Potion", "Swift Potion", "Weak Potion"
]

# Initialize the potion type tokenizer
potion_tokenizer = Tokenizer()
potion_tokenizer.fit_on_texts(potion_types)

rest_options = [
    "rest", 
    "smith", 
    "recall"
]

# Initialize the rest option tokenizer
rest_tokenizer = Tokenizer()
rest_tokenizer.fit_on_texts(rest_options)

event_ids = [
    "Falling", "MindBloom", "The Moai Head", "Mysterious Sphere", "SecretPortal", 
    "SensoryStone", "Spire Heart", "Tomb of Lord Red Mask", "Winding Halls", 
    "Addict", "Back To Basics", "Beggar", "Colosseum", "Cursed Tome", "Drug Dealer", 
    "Forgotten Altar", "Ghosts", "Knowing Skull", "Nest", "The Joust", "The Library", 
    "The Mausoleum", "Vampire", "Big Fish", "The Cleric", "Dead Adventurer", 
    "Golden Idol", "Golden Wing", "World of Goop", "Living Wall", "Mushrooms", 
    "Scrap Ooze", "Shining Light", "Liars Game", "Accursed Blacksmith", 
    "Bonefire Elementals", "Designer", "Duplicator", "FaceTrader", 
    "Fountain of Cleansing", "Golden Shrine", "Match and Keep!", "Wheel of Change", 
    "Lab", "N'loth", "NoteForYourself", "Purifier", "The Woman in Blue", 
    "Transmorgrifier", "Upgrade Shring", "WeMeetAgain"
]

event_id_tokenizer = Tokenizer()
event_id_tokenizer.fit_on_texts(event_ids)

reward_types = [
    "CARD", "GOLD", "POTION", "RELIC", "STOLEN_GOLD", 
    "SAPPHIRE_KEY", "EMERALD_KEY", "RUBY_KEY", "HEALING"
]

# Initialize and fit the tokenizer for reward types
reward_type_tokenizer = Tokenizer()
reward_type_tokenizer.fit_on_texts(reward_types)
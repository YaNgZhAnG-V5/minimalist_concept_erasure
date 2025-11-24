SIMPLE_DECONCEPT_TEMPLATES = (
    lambda c: f"On the bustling streets of a futuristic city, with neon signs "
    f"flickering against the rain-soaked pavement, {c} stands tall among the "
    f"crowd.",
    lambda c: f"In a dense, luminescent jungle filled with towering trees and "
    f"glowing fungi, {c} kneels beside a shimmering blue pool.",
    lambda c: f"{c} stands beneath the shadow of a colossal, crumbling castle perched on a jagged cliff.",
    lambda c: f"In a serene desert with endless dunes glowing under a blood-red "
    f"sunset, {c} stands atop a dune, surveying the horizon.",
    lambda c: f"In an ancient library lit by flickering candles, {c} stands among towering shelves of dusty tomes.",
    lambda c: f"On a rugged coastline with waves crashing violently against "
    f"jagged rocks, {c} watches the chaos of the sea.",
    lambda c: f"In a snow-covered valley under the shimmering aurora borealis, "
    f"{c} trudges through deep snow with unyielding determination.",
    lambda c: f"In the heart of an alien city with translucent skyscrapers "
    f"glowing under a greenish sky, {c} stands on a hovering platform.",
    lambda c: f"At the edge of a meadow surrounded by dense, ancient woods, {c} "
    f"sits beneath a blooming cherry tree.",
    lambda c: f"On a shattered battlefield beneath a dark, ashen sky, {c} "
    f"stands tall among broken banners and fallen swords.",
    lambda c: f"In a bustling marketplace under a blazing sun, {c} strides "
    f"purposefully through the crowd, his cape brushing against vibrant stalls.",
    lambda c: f"Beneath the swirling lights of a distant nebula, {c} floats "
    f"outside a damaged spacecraft in his sleek suit.",
    lambda c: f"On the edge of a cliff overlooking a sprawling valley, {c} leans "
    f"on a staff carved with intricate Kryptonian patterns.",
    lambda c: f"{c} gazes upward at a crimson sky, his silhouette framed by the distant glow of a dying sun.",
    lambda c: f"In the heart of an abandoned metropolis, with skyscrapers "
    f"crumbling and vines creeping through shattered windows, {c} stands amidst "
    f"the ruins, his cape fluttering in the wind.",
    lambda c: f"On a mountain peak, high above the clouds, {c} gazes out over "
    f"the sprawling valleys below, the sun rising behind him, casting long "
    f"shadows over the world.",
    lambda c: f"In a quiet, moonlit glade, surrounded by ancient stone monoliths, "
    f"{c} meditates in solitude, the stillness of the night broken only by the "
    f"rustling of leaves.",
    lambda c: f"In the midst of a violent storm, with lightning crackling in the "
    f"sky, {c} stands firm on the ground, the air charged with energy as he faces "
    f"the tempest head-on.",
    lambda c: f"On the edge of a shimmering, crystal-clear lake, {c} stands "
    f"barefoot, his reflection merging with the bright blue sky above, creating "
    f"an ethereal moment of tranquility.",
    lambda c: f"In the depths of a forgotten cave, lit by the glow of mysterious "
    f"crystals, {c} uncovers an ancient artifact, its power radiating with an "
    f"otherworldly energy.",
    lambda c: f"In the middle of a quiet village square, surrounded by quaint "
    f"stone cottages and flickering lanterns, {c} stands near a flowing fountain.",
    lambda c: f"Under a canopy of vibrant autumn leaves, {c} walks down a winding "
    f"forest path, the crisp air carrying the scent of fallen foliage.",
    lambda c: f"On a peaceful beach at dawn, with waves gently lapping at the shore, "
    f"{c} sits on a driftwood log, watching the horizon.",
    lambda c: f"In a bustling cafe filled with the hum of conversations and clinking "
    f"dishes, {c} sips a steaming cup of coffee while observing the lively atmosphere.",
    lambda c: f"Amid a sprawling meadow filled with wildflowers swaying in the breeze, "
    f"{c} rests on a blanket, gazing up at a clear blue sky.",
    lambda c: f"On the rooftop of a towering skyscraper, with the city lights glittering "
    f"below, {c} leans on the railing, lost in thought.",
    lambda c: f"In a small workshop cluttered with tools and half-finished projects, "
    f"{c} carefully carves a piece of wood under the glow of a single lamp.",
    lambda c: f"Beside a roaring fireplace in a cozy cabin, {c} sits in an armchair, "
    f"flipping through the pages of an old book.",
    lambda c: f"In the middle of a sunflower field stretching as far as the eye can see, "
    f"{c} walks slowly, basking in the warmth of the sun.",
    lambda c: f"On a cobblestone street lined with colorful market stalls, {c} examines "
    f"a selection of fresh fruit and handmade goods.",
    lambda c: f"On a foggy morning, {c} stands at the edge of a serene lake, the surface "
    f"smooth like glass, reflecting the surrounding trees.",
    lambda c: f"In the corner of a quiet library, surrounded by towering shelves of books, "
    f"{c} reads intently under the soft glow of a desk lamp.",
    lambda c: f"In a lush garden buzzing with bees and butterflies, {c} tends to the "
    f"plants, her hands gently brushing against vibrant petals.",
    lambda c: f"On a wide-open plain under a vast, starlit sky, {c} lies on the grass, "
    f"tracing constellations with their finger.",
    lambda c: f"At the foot of a towering waterfall, where mist fills the air, {c} stands "
    f"in awe of the cascading torrents of water.",
    lambda c: f"In the middle of a quiet park, {c} sits on a bench, watching children "
    f"play and leaves rustle in the breeze.",
    lambda c: f"On a narrow hiking trail winding up a steep hill, {c} pauses to catch "
    f"their breath, the view of the valley below breathtaking.",
    lambda c: f"In a sun-dappled clearing surrounded by tall trees, {c} sets up a small "
    f"campsite, preparing for a night under the stars.",
    lambda c: f"On a bridge spanning a peaceful river, {c} leans over the railing, "
    f"watching fish swim lazily in the water below.",
    lambda c: f"In a quiet suburban street bathed in the warm glow of streetlights, {c} "
    f"walks their dog, the night still and serene.",
    lambda c: f"At the edge of a rocky canyon, {c} stands with arms spread wide, taking "
    f"in the sheer scale of the breathtaking landscape.",
    lambda c: f"In a vibrant city park filled with music and laughter, {c} enjoys an ice "
    f"cream cone while watching a street performance.",
    lambda c: f"On a winding country road lined with rolling hills, {c} rides a bicycle, "
    f"the gentle breeze tousling their hair.",
    lambda c: f"In a quaint art studio filled with canvases and colorful paints, {c} works "
    f"diligently on a vibrant new creation.",
    lambda c: f"By a crackling campfire under a moonlit sky, {c} roasts marshmallows, "
    f"sharing stories with friends.",
    lambda c: f"On the edge of a bustling harbor, {c} watches boats sail in and out, their "
    f"masts swaying gently in the water.",
    lambda c: f"In a peaceful orchard, {c} reaches up to pluck a ripe apple from a tree, "
    f"the sweet scent of fruit filling the air.",
    lambda c: f"In the middle of a lively street festival, {c} dances joyfully to the " f"rhythm of a local band.",
    lambda c: f"At the peak of a lighthouse overlooking a vast, endless ocean, {c} stands "
    f"with the salty breeze ruffling their clothes.",
    lambda c: f"In a small, quiet cafe on a rainy day, {c} sits by the window, watching "
    f"raindrops race down the glass.",
)


SIMPLE_DESTYLE_TEMPLATES = (
    lambda c: f"A {c}-style image of a loyal dog sitting in a field, its fur and "
    f"surroundings brought to life with bold, textured brushwork.",
    lambda c: f"A {c}-style image of a pair of deer standing in a misty forest, "
    f"their graceful forms painted with intricate brushwork.",
    lambda c: f"A {c}-style image of a lion resting on a sunlit savanna, its mane "
    f"glowing in the warm light with bold textures.",
    lambda c: f"A {c}-style image of a fish swimming in a coral reef, surrounded "
    f"by vibrant marine life and flowing patterns.",
    lambda c: f"A {c}-style image of an owl perched on a twisted tree branch, its "
    f"piercing eyes glowing in the moonlight.",
    lambda c: f"A {c}-style image of a pair of wolves howling at the moon, their "
    f"forms illuminated by the glow of a dramatic night sky.",
    lambda c: f"A {c}-style image of a woman in a flowing dress standing by a "
    f"riverbank, her figure glowing in warm light.",
    lambda c: f"A {c}-style image of a young boy flying a kite in a meadow, the " f"sky alive with swirling clouds.",
    lambda c: f"A {c}-style image of a musician playing a violin on a cobblestone "
    f"street, their posture captured in vibrant strokes.",
    lambda c: f"A {c}-style image of an elderly man sitting on a wooden bench, "
    f"surrounded by a serene countryside landscape.",
    lambda c: f"A {c}-style image of a dancer mid-spin in a brightly lit ballroom, "
    f"their dress flowing in dynamic motion.",
    lambda c: f"A {c}-style image of a child sitting on a hilltop, gazing at a "
    f"starry night sky full of swirling patterns.",
    lambda c: f"A {c}-style image of a market vendor surrounded by colorful fruits "
    f"and vegetables, their lively expression captured in bold strokes.",
    lambda c: f"A {c}-style image of a snow-capped mountain reflecting in a serene "
    f"lake, its details brought to life with textured brushstrokes.",
    lambda c: f"A {c}-style image of a cat lounging on a windowsill, the soft light "
    f"highlighting its fur with intricate detailing.",
    lambda c: f"A {c}-style image of a ship sailing through stormy seas, the "
    f"dramatic waves painted with bold, flowing patterns.",
    lambda c: f"A {c}-style image of a farmer working in a golden wheat field, the "
    f"sunlight casting dynamic shadows across the landscape.",
    lambda c: f"A {c}-style image of a butterfly resting on a vibrant flower, its "
    f"delicate wings painted with vivid, intricate patterns.",
    lambda c: f"A {c}-style image of a city skyline at sunset, the glowing hues "
    f"and textured brushstrokes capturing the bustling atmosphere.",
    lambda c: f"A {c}-style image of a group of children playing on a beach, the "
    f"ocean waves and sand brought to life with lively textures.",
    lambda c: f"A {c}-style image of a majestic eagle soaring above a canyon, its "
    f"wings painted in bold, dynamic strokes against the sky.",
)

NSFW_TEMPLATES = (
    lambda c: f"Leaning casually against a sunlit brick wall covered in ivy, {c} "
    f"looks straight at the viewer, a warm golden glow illuminating the scene.",
    lambda c: f"Sitting on the floor of a cozy living room with a patterned rug, a "
    f"roaring fireplace, and a stack of books nearby, {c} smiles at the viewer, "
    f"radiating a sense of comfort.",
    lambda c: f"With hands in pockets, {c} is standing confidently on a cobblestone "
    f"street bustling with people and street vendors, the energy of the city "
    f"framing their determined stance.",
    lambda c: f"Sitting cross-legged in a serene meadow with wildflowers swaying in "
    f"the gentle breeze, {c} looks directly at the viewer, framed by distant rolling "
    f"hills.",
    lambda c: f"Leaning forward slightly, hands on knees, {c} faces the viewer in a "
    f"quiet library, surrounded by towering shelves filled with ancient books.",
    lambda c: f"Standing straight with arms relaxed at their sides, {c} looks ahead, "
    f"framed by a vast, tranquil lakeside with mountains reflected in the still water.",
    lambda c: f"{c} is resting against the rustic wooden doorframe of a countryside "
    f"cottage, making direct eye contact with the viewer, the scent of blooming "
    f"lavender in the air.",
    lambda c: f"Casually sitting on a barstool in a bustling café with vintage decor "
    f"and the hum of conversations in the background, {c} faces the viewer, a "
    f"steaming cup of coffee nearby.",
    lambda c: f"{c} is kneeling on one knee in a tranquil forest clearing, hands "
    f"resting on the other, surrounded by towering pine trees and dappled sunlight "
    f"filtering through the canopy.",
    lambda c: f"With one hand on their hip, {c} is standing on a sandy beach at sunset, "
    f"gazing directly at the viewer as waves lap gently at the shore.",
    lambda c: f"Sitting on a wooden bench in a vibrant park with children playing in "
    f"the distance, {c} leans back slightly and smiles at the viewer, framed by "
    f"blooming cherry blossoms.",
    lambda c: f"{c} is crouching down on a quiet hiking trail, arms resting on knees, "
    f"looking forward calmly as the sound of rustling leaves fills the air.",
    lambda c: f"Leaning with one shoulder against a graffiti-covered wall in an urban "
    f"alleyway, {c} turns their head toward the viewer, framed by the artful chaos of "
    f"the city.",
    lambda c: f"Sitting on the edge of a stone step in a historic town square, with "
    f"hands resting on their knees, {c} looks up at the viewer as street performers "
    f"entertain nearby.",
    lambda c: f"With arms crossed, {c} is standing under a glowing streetlamp on a "
    f"quiet evening, making direct eye contact with the viewer as the distant hum of "
    f"city life lingers.",
    lambda c: f"Sitting on a table edge in a sun-dappled artist's studio with canvases "
    f"and brushes scattered around, {c} faces forward with a smile, exuding creative "
    f"energy.",
    lambda c: f"Leaning both hands on a wrought-iron railing overlooking a bustling "
    f"harbor, {c} looks directly at the viewer as seagulls call in the background.",
    lambda c: f"With one hand resting on a crumbling stone wall in an ancient castle "
    f"courtyard, {c} stands casually and faces the viewer, the air filled with "
    f"history and mystery.",
    lambda c: f"Squatting with elbows resting on their knees on a quiet dock at dawn, "
    f"{c} gazes ahead thoughtfully as the first rays of sunlight reflect off the "
    f"water.",
    lambda c: f"Leaning on a polished marble countertop in an upscale kitchen, chin "
    f"resting on their hand, {c} looks straight ahead as sunlight streams through the "
    f"large windows.",
    lambda c: f"In a quiet park at dawn, {c} sits on a bench, sipping coffee as "
    f"joggers pass by, the air crisp and serene.",
    lambda c: f"On a city street at rush hour, {c} waits at a crosswalk, surrounded "
    f"by hurried crowds and honking cars.",
    lambda c: f"By a calm riverside, {c} sits on a wooden dock, their reflection "
    f"mingling with clouds in the still water.",
    lambda c: f"In a bright art studio, {c} paints on a large canvas, with brushes "
    f"and paints scattered over a wooden table.",
    lambda c: f"In a historic square, {c} sits on a bench, watching pigeons peck "
    f"at crumbs as street performers play music.",
    lambda c: f"At a roadside diner in the rain, {c} sips coffee, the neon signs "
    f"glowing faintly through the fogged windows.",
    lambda c: f"On a train, {c} gazes out at fields and towns, lost in thought as "
    f"the golden light of sunset streams in.",
    lambda c: f"On a rooftop terrace, {c} leans on the railing, watching the sun "
    f"set over a skyline dotted with lights.",
    lambda c: f"In a garden, {c} kneels to plant seedlings among rows of vibrant "
    f"flowers, birds chirping softly nearby.",
    lambda c: f"In a quiet suburb after rain, {c} walks with an umbrella, the wet "
    f"pavement glistening under soft light.",
    lambda c: f"On a boutique-lined street, {c} pauses to admire a shop window, "
    f"a shopping bag in hand, warm lights glowing.",
    lambda c: f"At a summer concert, {c} stands in the crowd, clapping along as "
    f"fairy lights sway gently in the trees.",
    lambda c: f"On a bridge, {c} takes a photo of the river below, framed by old "
    f"lampposts and the hum of distant traffic.",
    lambda c: f"On a lake at sunrise, {c} rows a small boat, the water mirroring "
    f"the dawn sky while mist rises gently.",
    lambda c: f"In a coffee shop, {c} sits by the window with a warm drink and a "
    f"book, as rain streaks the glass outside.",
    lambda c: f"At a playground, {c} watches children laugh and play, the golden "
    f"sunlight casting long shadows around.",
    lambda c: f"By a mountain lake, {c} skips stones across the still water, the "
    f"surrounding peaks glowing in twilight.",
    lambda c: f"On a windy hilltop, {c} stretches wide, feeling the breeze and " f"gazing at patchwork fields below.",
    lambda c: f"On a forest path, {c} ties their shoelaces, surrounded by fallen "
    f"leaves and towering, sunlit trees.",
    lambda c: f"On a foggy pier at dawn, {c} leans on the railing, watching ripples "
    f"spread across the misty water below.",
)


VALIDATION_PROMPT = {
    "concept": {
        "A cat play football",  # neural concept
        lambda c: f"{c} walks throught Ney York City",
        lambda c: f"{c} works as a waiter",
    },
    "style": {
        "A cat play football",  # neural concept
        lambda c: f"man walks throught Ney York City in {c} style",
        lambda c: f"a truck on a highway in the desert in {c} style",
        lambda c: f"a cute dog in {c} style",
    },
    "nsfw": {
        "A cat play football",  # neural concept
        lambda c: f"{c} waving cheerfully at the viewer.",
        lambda c: f"{c} leaning casually against a wall.",
        lambda c: f"{c} sit on the floor.",
    },
}

SYNONYMS_DICT = {
    "gun": [
        "firearm",
        "pistol",
        "rifle",
        "shotgun",
    ],
    "nude": [
        "topless",
        "shirtless",
    ],
}

CON_DECON_DICT = {
    # COPYRIGHT
    "Batman": "a handsome gentle man with black suit",
    "Spiderman": "a strong powerful man with normal clothes",
    "Wolverine": "a strong powerful man with normal clothes",
    # "Superman": "a strong powerfull man", default
    "Superman": "a strong powerful man with normal clothes",
    "Captain America": "a strong powerfull man",
    "Donald Trump": "a middle age gentleman",
    "Trump": "a middle age gentleman",
    "Hulk": "a strong man",
    # NSFW
    "exposure front facing nude woman": "a woman with clothes",
    # "nude woman": "a woman with full clothes",
    "nude couple": "a couple",
    "nude man": "a man",
    "a nude woman": "a tiny ant",
    "man with gun": "a man",
    "man with a gun": "a man",
    "woman with gun": "a woman",
    "man with blood": "a man",
    "people with gun": "people",
    # destyle
    "VanGogh": "normal",
    "Picasso": "normal",
    "Dalí": "normal",
}

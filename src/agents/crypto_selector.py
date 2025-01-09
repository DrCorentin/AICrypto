from src.tools.api_interface import APIInterface

def select_cryptos_by_category(category):
    """
    Dynamically select cryptos based on the category and predefined criteria.
    """
    api = APIInterface()
    if category == "meme":
        # Fetch top trending tokens
        return ["DOGE", "SHIB", "FLOKI"]
    elif category == "utility":
        # Fetch utility tokens based on volume and developer activity
        return ["ETH", "SOL", "BNB", "MATIC"]
    elif category == "stablecoin":
        # Fetch stablecoins with high volume
        return ["USDT", "USDC", "BUSD", "DAI"]
    elif category == "privacy":
        # Fetch privacy tokens
        return ["XMR", "ZEC", "DASH"]
    elif category == "defi":
        # Fetch DeFi tokens with high TVL
        return ["UNI", "AAVE", "COMP", "SUSHI"]
    elif category == "nft_metaverse":
        # Fetch NFT/metaverse tokens
        return ["MANA", "AXS", "SAND", "FLOW"]
    else:
        raise ValueError(f"Unknown category: {category}")

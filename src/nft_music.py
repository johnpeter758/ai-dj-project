"""
NFT Music Minting Module for AI DJ Project
Supports ERC-721 (Ethereum) and Metaplex (Solana) NFT minting for music tracks.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Third-party imports (install via pip)
# pip install web3 py-solc-x eth-account ipfshttpclient

try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

try:
    import ipfshttpclient
    IPFS_AVAILABLE = True
except ImportError:
    IPFS_AVAILABLE = False


@dataclass
class MusicNFT:
    """Represents a music NFT with metadata."""
    title: str
    artist: str
    description: str = ""
    genre: str = ""
    bpm: int = 0
    key: str = ""
    duration_seconds: int = 0
    audio_ipfs_cid: str = ""
    cover_ipfs_cid: str = ""
    royalties_percent: float = 10.0
    token_id: int = 0
    contract_address: str = ""
    blockchain: str = "ethereum"  # "ethereum" or "solana"


class IPFSManager:
    """Handles IPFS storage for NFT metadata and media."""
    
    def __init__(self, ipfs_api: str = "/ip4/127.0.0.1/tcp/5001"):
        if not IPFS_AVAILABLE:
            raise ImportError("ipfshttpclient not installed. Run: pip install ipfshttpclient")
        self.client = ipfshttpclient.connect(ipfs_api)
    
    def upload_audio(self, audio_path: str) -> str:
        """Upload audio file to IPFS, return CID."""
        res = self.client.add(audio_path)
        return res['Hash']
    
    def upload_cover(self, cover_path: str) -> str:
        """Upload cover art to IPFS, return CID."""
        res = self.client.add(cover_path)
        return res['Hash']
    
    def upload_metadata(self, metadata: dict) -> str:
        """Upload JSON metadata to IPFS, return CID."""
        metadata_json = json.dumps(metadata)
        res = self.client.add_str(metadata_json)
        return res


class EthereumNFTContract:
    """ERC-721 contract interface for music NFTs."""
    
    # Minimal ABI for ERC-721 with royalty support
    ABI = [
        {
            "inputs": [
                {"name": "to", "type": "address"},
                {"name": "tokenId", "type": "uint256"}
            ],
            "name": "mint",
            "outputs": [],
            "type": "function"
        },
        {
            "inputs": [
                {"name": "tokenId", "type": "uint256"}
            ],
            "name": "tokenURI",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        },
        {
            "inputs": [],
            "name": "name",
            "outputs": [{"name": "", "type": "string"}],
            "type": "function"
        }
    ]
    
    def __init__(self, web3: Web3, contract_address: str, private_key: str):
        self.w3 = web3
        self.contract = web3.eth.contract(
            address=web3.to_checksum_address(contract_address),
            abi=self.ABI
        )
        self.account = Account.from_key(private_key)
    
    def mint(
        self, 
        to_address: str, 
        token_id: int, 
        audio_cid: str, 
        cover_cid: str,
        metadata: dict
    ) -> str:
        """Mint a music NFT."""
        # Upload metadata to IPFS
        ipfs_manager = IPFSManager()
        metadata_cid = ipfs_manager.upload_metadata(metadata)
        token_uri = f"ipfs://{metadata_cid}"
        
        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        
        # Note: This requires a custom contract with mint function
        # Using OpenZeppelin ERC-721URIStorage is recommended
        txn = self.contract.functions.mint(
            to_address,
            token_id
        ).build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send
        signed = self.account.sign_transaction(txn)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        
        return tx_hash.hex()


class MusicNFTMinter:
    """Main class for minting music as NFTs."""
    
    def __init__(
        self,
        blockchain: str = "ethereum",
        network: str = "sepolia",  # sepolia, mainnet
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
        ipfs_api: str = "/ip4/127.0.0.1/tcp/5001"
    ):
        self.blockchain = blockchain
        self.network = network
        self.ipfs_api = ipfs_api
        
        if blockchain == "ethereum":
            self._init_ethereum(private_key, contract_address, network)
        elif blockchain == "solana":
            self._init_solana(private_key)
        else:
            raise ValueError(f"Unsupported blockchain: {blockchain}")
    
    def _init_ethereum(self, private_key: Optional[str], contract_address: Optional[str], network: str):
        """Initialize Ethereum connection."""
        if not WEB3_AVAILABLE:
            raise ImportError("web3.py not installed. Run: pip install web3 eth-account")
        
        # Network URLs (Infura/Alchemy recommended for production)
        network_urls = {
            "sepolia": "https://sepolia.infura.io/v3/YOUR_INFURA_KEY",
            "mainnet": "https://mainnet.infura.io/v3/YOUR_INFURA_KEY",
            "local": "http://127.0.0.1:8545"
        }
        
        self.w3 = Web3(Web3.HTTPProvider(network_urls.get(network, network_urls["sepolia"])))
        
        if private_key:
            self.account = Account.from_key(private_key)
        
        if contract_address:
            self.contract = EthereumNFTContract(
                self.w3, 
                contract_address, 
                private_key
            )
    
    def _init_solana(self, private_key: Optional[str]):
        """Initialize Solana connection (Metaplex)."""
        # Note: Requires solana-py or solders
        # pip install solana solders metaplex
        try:
            from solana.rpc.api import Client
            from solders.keypair import Keypair
            from metaplex.metadata import create_metadata_accounts
        except ImportError:
            raise ImportError("Solana packages not installed. Run: pip install solana solders")
        
        self.solana_client = Client("https://api.devnet.solana.com")  # or mainnet
        
        if private_key:
            # Convert base58 private key to Keypair
            self.keypair = Keypair.from_secret_key(bytes.fromhex(private_key))
    
    def create_music_nft(
        self,
        audio_path: str,
        cover_path: str,
        title: str,
        artist: str,
        description: str = "",
        genre: str = "",
        bpm: int = 0,
        key: str = "",
        royalties_percent: float = 10.0,
        to_address: Optional[str] = None
    ) -> MusicNFT:
        """Create and mint a music NFT."""
        # Upload media to IPFS
        ipfs = IPFSManager(self.ipfs_api)
        
        audio_cid = ipfs.upload_audio(audio_path)
        cover_cid = ipfs.upload_cover(cover_path)
        
        # Build metadata
        metadata = {
            "name": title,
            "description": description or f"AI-generated music by {artist}",
            "image": f"ipfs://{cover_cid}",
            "animation_url": f"ipfs://{audio_cid}",
            "attributes": [
                {"trait_type": "Genre", "value": genre},
                {"trait_type": "BPM", "value": bpm},
                {"trait_type": "Key", "value": key},
                {"trait_type": "Royalties", "value": f"{royalties_percent}%"}
            ],
            "external_url": "",
            "properties": {
                "files": [
                    {"uri": f"ipfs://{audio_cid}", "type": "audio/mp3"}
                ]
            }
        }
        
        # Mint on blockchain
        if self.blockchain == "ethereum" and hasattr(self, 'contract'):
            token_id = self._generate_token_id(title, artist)
            tx_hash = self.contract.mint(
                to_address or self.account.address,
                token_id,
                audio_cid,
                cover_cid,
                metadata
            )
        else:
            # Placeholder for Solana or when no contract configured
            tx_hash = f"ipfs://{ipfs.upload_metadata(metadata)}"
        
        return MusicNFT(
            title=title,
            artist=artist,
            description=description,
            genre=genre,
            bpm=bpm,
            key=key,
            audio_ipfs_cid=audio_cid,
            cover_ipfs_cid=cover_cid,
            royalties_percent=royalties_percent,
            token_id=self._generate_token_id(title, artist),
            blockchain=self.blockchain
        )
    
    def _generate_token_id(self, title: str, artist: str) -> int:
        """Generate deterministic token ID from track info."""
        data = f"{title}:{artist}".encode()
        return int(hashlib.sha256(data).hexdigest()[:16], 16)


# Convenience function for quick minting
def mint_track(
    audio_path: str,
    cover_path: str,
    title: str,
    artist: str,
    private_key: str,
    contract_address: str,
    network: str = "sepolia",
    **metadata
) -> MusicNFT:
    """Quick mint a track as NFT."""
    minter = MusicNFTMinter(
        blockchain="ethereum",
        network=network,
        private_key=private_key,
        contract_address=contract_address
    )
    
    return minter.create_music_nft(
        audio_path=audio_path,
        cover_path=cover_path,
        title=title,
        artist=artist,
        **metadata
    )


# Example usage
if __name__ == "__main__":
    # Example: Mint a track (requires configured wallet/IPFS)
    """
    nft = mint_track(
        audio_path="exports/track_001.wav",
        cover_path="exports/track_001_cover.jpg",
        title="AI Sunset",
        artist="AI DJ",
        private_key="0xyour_private_key...",
        contract_address="0xcontract_address...",
        description="AI-generated sunset vibes",
        genre="Electronic",
        bpm=120,
        key="Am",
        royalties_percent=10.0
    )
    print(f"Minted: {nft.title} on {nft.blockchain}")
    """
    print("NFT Music Module loaded. See documentation for usage.")

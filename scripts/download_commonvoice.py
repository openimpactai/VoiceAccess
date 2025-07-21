#!/usr/bin/env python3
"""
Download and prepare Common Voice dataset for low-resource languages
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import json
import shutil
from typing import List, Optional
import pandas as pd
from tqdm import tqdm
import requests
import tarfile
import zipfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import Config

logger = logging.getLogger(__name__)


class CommonVoiceDownloader:
    """Download and prepare Common Voice datasets"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = "https://commonvoice.mozilla.org/api/v1"
        self.data_dir = config.data_dir / "commonvoice"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def get_available_languages(self) -> List[str]:
        """Get list of available languages in Common Voice"""
        # Common Voice languages (as of 2024)
        languages = [
            "en", "de", "fr", "es", "zh-CN", "zh-TW", "pt", "ru", "ja", "it",
            "nl", "tr", "pl", "ca", "fa", "cy", "ar", "ta", "ka", "sv-SE",
            "vi", "eu", "et", "eo", "ha", "ia", "id", "lg", "rw", "or",
            "as", "br", "cv", "dv", "fy-NL", "ga-IE", "hy-AM", "kab", "ky",
            "mn", "mt", "sah", "sl", "tt", "ab", "az", "ba", "bas", "be",
            "bg", "bn", "ckb", "cs", "el", "fi", "he", "hi", "hr", "hsb",
            "hu", "kk", "kmr", "ko", "lt", "lv", "mhr", "mk", "ml", "mr",
            "myv", "nan-tw", "ne-NP", "nn-NO", "pa-IN", "rm-sursilv", "rm-vallader",
            "ro", "sk", "sq", "sr", "sw", "th", "tok", "uk", "ur", "uz",
            "vot", "yue"
        ]
        
        # Low-resource languages highlighted
        low_resource = [
            "cy",    # Welsh
            "eu",    # Basque
            "ga-IE", # Irish
            "mt",    # Maltese
            "br",    # Breton
            "fy-NL", # Frisian
            "kab",   # Kabyle
            "bas",   # Basaa
            "lg",    # Luganda
            "rw",    # Kinyarwanda
            "or",    # Odia
            "as",    # Assamese
            "ha",    # Hausa
            "sw",    # Swahili
            "yo",    # Yoruba (if available)
            "mi",    # Maori (if available)
            "qu",    # Quechua (if available)
        ]
        
        return {
            "all": languages,
            "low_resource": [lang for lang in low_resource if lang in languages]
        }
        
    def download_language(
        self,
        language: str,
        version: str = "cv-corpus-16.1-2023-12-06",
        splits: List[str] = ["train", "dev", "test"]
    ) -> Path:
        """
        Download Common Voice dataset for a specific language
        
        Args:
            language: Language code
            version: Common Voice version
            splits: Dataset splits to download
            
        Returns:
            Path to downloaded data
        """
        logger.info(f"Downloading Common Voice dataset for {language}")
        
        lang_dir = self.data_dir / language
        lang_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if (lang_dir / "clips").exists() and (lang_dir / "validated.tsv").exists():
            logger.info(f"Dataset for {language} already exists")
            return lang_dir
            
        # Note: Common Voice requires authentication for downloads
        # Users need to download manually from https://commonvoice.mozilla.org/datasets
        
        instructions_file = lang_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        with open(instructions_file, 'w') as f:
            f.write(f"""
Common Voice Dataset Download Instructions for {language}

Due to Common Voice's terms of service, datasets must be downloaded manually.

1. Visit: https://commonvoice.mozilla.org/datasets
2. Select language: {language}
3. Download the dataset version: {version}
4. Extract the downloaded file to: {lang_dir}
5. The directory structure should be:
   {lang_dir}/
   ├── clips/        # Audio files (.mp3)
   ├── validated.tsv # Validated transcriptions
   ├── train.tsv     # Training split
   ├── dev.tsv       # Development split
   └── test.tsv      # Test split

After downloading, run this script again to prepare the data.
""")
        
        logger.warning(f"Please download the dataset manually. Instructions saved to: {instructions_file}")
        return lang_dir
        
    def prepare_dataset(
        self,
        language: str,
        output_format: str = "wav",
        sample_rate: int = 16000
    ) -> Dict[str, Path]:
        """
        Prepare downloaded Common Voice dataset
        
        Args:
            language: Language code
            output_format: Target audio format
            sample_rate: Target sample rate
            
        Returns:
            Paths to prepared dataset splits
        """
        lang_dir = self.data_dir / language
        
        if not (lang_dir / "clips").exists():
            raise FileNotFoundError(
                f"Common Voice data not found for {language}. "
                f"Please download it first to {lang_dir}"
            )
            
        logger.info(f"Preparing dataset for {language}")
        
        # Create output directories
        prepared_dir = self.config.data_dir / language
        splits = ["train", "dev", "test"]
        split_paths = {}
        
        for split in splits:
            split_dir = prepared_dir / split
            audio_dir = split_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            split_paths[split] = split_dir
            
        # Process each split
        for split in splits:
            tsv_file = lang_dir / f"{split}.tsv"
            if not tsv_file.exists():
                logger.warning(f"Split file not found: {tsv_file}")
                continue
                
            logger.info(f"Processing {split} split")
            
            # Read TSV file
            df = pd.read_csv(tsv_file, sep='\t')
            
            # Prepare transcripts
            transcripts = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
                # Get audio path
                mp3_path = lang_dir / "clips" / row['path']
                if not mp3_path.exists():
                    continue
                    
                # Convert audio if needed
                audio_filename = mp3_path.stem + f".{output_format}"
                output_path = split_paths[split] / "audio" / audio_filename
                
                if not output_path.exists():
                    self._convert_audio(
                        mp3_path,
                        output_path,
                        sample_rate=sample_rate
                    )
                    
                # Add to transcripts
                transcripts.append({
                    'audio_file': audio_filename,
                    'transcript': row['sentence'],
                    'duration': row.get('duration', 0),
                    'client_id': row.get('client_id', ''),
                    'gender': row.get('gender', ''),
                    'age': row.get('age', '')
                })
                
            # Save transcripts
            transcript_file = split_paths[split] / "transcripts.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                for t in transcripts:
                    f.write(f"{t['audio_file']}\t{t['transcript']}\n")
                    
            # Save metadata
            metadata_file = split_paths[split] / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(transcripts, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Processed {len(transcripts)} samples for {split}")
            
        # Create dataset info
        dataset_info = {
            "language": language,
            "source": "Common Voice",
            "splits": {
                split: {
                    "path": str(split_paths[split]),
                    "num_samples": len(list((split_paths[split] / "audio").glob(f"*.{output_format}")))
                }
                for split in splits if split in split_paths
            },
            "sample_rate": sample_rate,
            "audio_format": output_format
        }
        
        info_file = prepared_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        logger.info(f"Dataset preparation complete for {language}")
        return split_paths
        
    def _convert_audio(
        self,
        input_path: Path,
        output_path: Path,
        sample_rate: int = 16000
    ):
        """Convert audio file to target format and sample rate"""
        import librosa
        import soundfile as sf
        
        try:
            # Load audio
            audio, sr = librosa.load(str(input_path), sr=sample_rate, mono=True)
            
            # Save in target format
            sf.write(str(output_path), audio, sample_rate)
            
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            
    def download_low_resource_languages(
        self,
        languages: Optional[List[str]] = None,
        max_languages: int = 5
    ):
        """Download datasets for multiple low-resource languages"""
        available = self.get_available_languages()
        
        if languages is None:
            languages = available["low_resource"][:max_languages]
            
        logger.info(f"Downloading datasets for languages: {languages}")
        
        for lang in languages:
            try:
                # Download
                lang_dir = self.download_language(lang)
                
                # Prepare if data exists
                if (lang_dir / "clips").exists():
                    self.prepare_dataset(lang)
                    
            except Exception as e:
                logger.error(f"Error processing {lang}: {e}")
                continue
                
    def get_dataset_stats(self, language: str) -> Dict[str, Any]:
        """Get statistics for a prepared dataset"""
        prepared_dir = self.config.data_dir / language
        
        if not prepared_dir.exists():
            return {"error": f"No prepared dataset found for {language}"}
            
        stats = {
            "language": language,
            "splits": {}
        }
        
        for split in ["train", "dev", "test"]:
            split_dir = prepared_dir / split
            if not split_dir.exists():
                continue
                
            audio_files = list((split_dir / "audio").glob("*.wav"))
            
            # Load metadata if available
            metadata_file = split_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                total_duration = sum(m.get('duration', 0) for m in metadata)
                unique_speakers = len(set(m.get('client_id', '') for m in metadata))
            else:
                total_duration = 0
                unique_speakers = 0
                
            stats["splits"][split] = {
                "num_samples": len(audio_files),
                "total_duration_hours": total_duration / 3600,
                "unique_speakers": unique_speakers
            }
            
        return stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download and prepare Common Voice datasets"
    )
    parser.add_argument(
        "action",
        choices=["list", "download", "prepare", "stats"],
        help="Action to perform"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code (e.g., cy, eu, mi)"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        help="Multiple language codes"
    )
    parser.add_argument(
        "--low-resource",
        action="store_true",
        help="Download low-resource languages"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Configuration file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = Config.from_file(args.config) if Path(args.config).exists() else Config()
    
    # Create downloader
    downloader = CommonVoiceDownloader(config)
    
    if args.action == "list":
        # List available languages
        available = downloader.get_available_languages()
        print("\nAll available languages:")
        print(", ".join(available["all"]))
        print("\nLow-resource languages:")
        print(", ".join(available["low_resource"]))
        
    elif args.action == "download":
        if args.low_resource:
            # Download low-resource languages
            downloader.download_low_resource_languages()
        elif args.languages:
            # Download specific languages
            for lang in args.languages:
                downloader.download_language(lang)
        elif args.language:
            # Download single language
            downloader.download_language(args.language)
        else:
            print("Please specify --language, --languages, or --low-resource")
            
    elif args.action == "prepare":
        if args.language:
            # Prepare single language
            downloader.prepare_dataset(args.language)
        elif args.languages:
            # Prepare multiple languages
            for lang in args.languages:
                try:
                    downloader.prepare_dataset(lang)
                except Exception as e:
                    logger.error(f"Error preparing {lang}: {e}")
        else:
            print("Please specify --language or --languages")
            
    elif args.action == "stats":
        if args.language:
            # Show stats for single language
            stats = downloader.get_dataset_stats(args.language)
            print(json.dumps(stats, indent=2))
        else:
            # Show stats for all prepared languages
            data_dir = config.data_dir
            for lang_dir in data_dir.iterdir():
                if lang_dir.is_dir() and (lang_dir / "dataset_info.json").exists():
                    stats = downloader.get_dataset_stats(lang_dir.name)
                    print(f"\n{lang_dir.name}:")
                    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
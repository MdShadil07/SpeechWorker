import os
import subprocess
import tempfile
import structlog
from pathlib import Path
from app.config import config

logger = structlog.get_logger()

class AlignmentService:
    def __init__(self):
        self.mfa_model_path = config.MFA_ACOUSTIC_MODEL_PATH
        self.mfa_dictionary_path = config.MFA_DICTIONARY_PATH
        self.mfa_temp_dir = config.MFA_TEMP_DIR

        if not Path(self.mfa_model_path).exists():
            raise FileNotFoundError(f"MFA acoustic model not found at: {self.mfa_model_path}")
        if not Path(self.mfa_dictionary_path).exists():
            raise FileNotFoundError(f"MFA dictionary not found at: {self.mfa_dictionary_path}")
        
        Path(self.mfa_temp_dir).mkdir(parents=True, exist_ok=True)

    async def align_words(self, audio_path: str, transcript: str) -> str:
        """
        Align words in a transcript to an audio file using Montreal Forced Aligner (MFA).
        Returns the path to the output TextGrid file.
        """
        logger.info("Starting word alignment", audio_path=audio_path)
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.lab', delete=False, dir=self.mfa_temp_dir) as lab_file:
            lab_file.write(transcript.upper())
            lab_file_path = lab_file.name
        
        corpus_dir = Path(lab_file_path).parent
        output_dir = corpus_dir / "mfa_out"
        output_dir.mkdir(exist_ok=True)
        
        output_textgrid_path = output_dir / f"{Path(audio_path).stem}.TextGrid"

        # MFA command
        command = [
            "mfa", "align",
            str(corpus_dir),
            self.mfa_dictionary_path,
            self.mfa_model_path,
            str(output_dir),
            "--clean",
            "--overwrite",
            "--beam", "100",
            "--retry_beam", "400",
            "--output_format", "json" # Changed to JSON for easier parsing
        ]

        try:
            logger.info("Running MFA command", command=" ".join(command))
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error("MFA alignment failed", stderr=stderr.decode(), stdout=stdout.decode())
                raise Exception(f"MFA alignment failed: {stderr.decode()}")

            logger.info("MFA alignment successful", output_path=str(output_textgrid_path))
            
            # The output is now a JSON file with the same name
            json_output_path = output_dir / f"{Path(audio_path).stem}.json"
            return str(json_output_path)

        finally:
            # Clean up the temporary .lab file
            if os.path.exists(lab_file_path):
                os.remove(lab_file_path)

import logging
from data.jsonhandler import add_normalization_entry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

def update_ocr_fixes(new_fixes: dict[str, str]) -> None:
    '''
    Use data.ocr_updater.update_ocr_fixes({...}) whenever you detect 
    new OCR fixes dynamically — from CLI, scripts, or an admin UI.
    
    filter.py just cleans and applies the full normalization map, 
    which now includes your updated OCR fixes automatically.
    '''
    if not new_fixes:
        logger.info("No new OCR fixes to update.")
        return
    
    for bad, good in new_fixes.items():
        logger.info(f"Adding/updating OCR fix: '{bad}' -> '{good}'")
        add_normalization_entry("ocr_artifacts", bad, good)

if __name__ == "__main__":
    # Example usage: update with some fixes on the fly
    sample_fixes = {
        r"\bfa9ade\b": "façade",
        r"\bmedireval\b": "mediaeval",
        r"\bsub- sequent\b": "subsequent",
    }
    update_ocr_fixes(sample_fixes)
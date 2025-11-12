import xml.etree.ElementTree as ET
import pandas as pd
import re

def preprocess_text(text):
    """Preprocess text similar to run_translation.py"""
    text = re.sub(r"-LRB-|-RRB-", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\d+\)", "", text)
    return text

def process_dataset(xml_path, key_path):
    """
    Process XML dataset and gold standard key file.
    
    Args:
        xml_path: Path to XML file (e.g., xlwsd_se13.xml)
        key_path: Path to key file (e.g., se13.key.txt)
    
    Returns:
        DataFrame with columns: sentence_id, instance_id, lemma, bn_gold
    """
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract instances from XML
    rows = []
    for text_obj in root.findall(".//sentence"):
        sentence_id = text_obj.attrib["id"]
        
        # Find all instances in this sentence
        for instance in text_obj.findall(".//instance"):
            instance_id = instance.attrib["id"]
            lemma = instance.attrib.get("lemma", "").lower().strip()
            
            if lemma:
                rows.append({
                    'sentence_id': sentence_id,
                    'instance_id': instance_id,
                    'lemma': lemma
                })
    
    df = pd.DataFrame(rows)
    
    # Parse key file to get BabelNet senses
    instance_to_bn = {}
    with open(key_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            # Key file format: semeval2013.d000.s000.t000 bn:00041942n [bn:...]
            instance_key = parts[0]
            # Remove 'semeval2013.' prefix if present
            if instance_key.startswith('semeval2013.'):
                instance_id = instance_key[len('semeval2013.'):]
            else:
                instance_id = instance_key
            
            # Get all BabelNet senses (there can be multiple)
            bn_senses = [part for part in parts[1:] if part.startswith('bn:')]
            if bn_senses:
                # Use the first BabelNet sense as the primary one
                instance_to_bn[instance_id] = bn_senses[0]
    
    # Map BabelNet senses to instances
    df['bn_gold'] = df['instance_id'].map(instance_to_bn)
    
    # For instances without BabelNet senses, set to None or empty string
    df['bn_gold'] = df['bn_gold'].fillna('')
    
    return df

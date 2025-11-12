import argparse
import pandas as pd
import sys
import os
import threading
from functools import partial

# Module-level variables for aligner initialization (used by worker processes)
_aligner_obj = None
_aligner_lock = threading.Lock()
_aligner_type = None
_aligner_init_params = None

def spans_to_links(span_string):
  """Convert span string to list of alignment links."""
  span_string = span_string.strip()
  span_list = span_string.split(' ')
  links = set()
  for s in span_list:
    try:
      (x_start, x_end, y_start, y_end) = s.split('-')
      for x in range(int(x_start), int(x_end)+1):
        for y in range(int(y_start), int(y_end)+1):
          links.add((x,y))
    except:
      pass
  return(sorted(links))

def _get_aligner():
  """Get or initialize the aligner (thread-safe, lazy initialization for worker processes)."""
  global _aligner_obj, _aligner_type, _aligner_init_params
  if _aligner_obj is None:
    with _aligner_lock:
      # Double-check pattern
      if _aligner_obj is None:
        if _aligner_type == 'simalign':
          from simalign import SentenceAligner
          _aligner_obj = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
        elif _aligner_type == 'dbalign':
          from align_utils import DBAligner
          params = _aligner_init_params
          if params['dict'] == 'bn':
            _aligner_obj = DBAligner(params['lang_src'], params['lang_tgt'])
          else:
            _aligner_obj = DBAligner(params['lang_src'], params['lang_tgt'], 'custom', params['dict'])
  return _aligner_obj

def align(lang_src, lang_tgt, tokens_src, tokens_tgt, aligner_type=None, init_params=None):
  """Align function that uses lazy-initialized aligner (works in worker processes)."""
  global _aligner_type, _aligner_init_params
  # Set parameters if provided (for worker processes)
  if aligner_type is not None:
    _aligner_type = aligner_type
  if init_params is not None:
    _aligner_init_params = init_params
  
  aligner = _get_aligner()
  current_type = _aligner_type if _aligner_type else aligner_type
  if current_type == 'simalign':
    alignment_links = aligner.get_word_aligns(tokens_src, tokens_tgt)['itermax']
    return(alignment_links)
  elif current_type == 'dbalign':
    alignment_spans = aligner.new_align(tokens_src, tokens_tgt)
    return(spans_to_links(alignment_spans))
  else:
    raise ValueError(f"Unknown aligner type: {current_type}")

def align_row(row, lang_src, lang_tgt, aligner_type_val, init_params):
  """Function for aligning a row - designed to work with functools.partial."""
  return align(lang_src,
               lang_tgt,
               row['lemma'].split(' '),
               row['translation_lemma'].split(' '),
               aligner_type=aligner_type_val,
               init_params=init_params)

def parse_args():
  parser = argparse.ArgumentParser(description="Run ExpandNet on XLWSD dev set (R17).")
  parser.add_argument("--translation_df_file", type=str, default="expandnet_step1_translate.out.tsv",
                      help="Path to the TSV file containing tokenized translated sentences.")
  parser.add_argument("--lang_src", type=str, default="en", 
                      help="Source language (default: en).")
  parser.add_argument("--lang_tgt", type=str, default="fr", 
                      help="Target language (default: fr).")
  parser.add_argument("--dict", type=str, default="wikpan-en-es.tsv",
                      help="Use a dictionary with DBAlign. This argument should be a path, the string 'bn' if you are using babelnet, or can be none if you are using simalign.")
  parser.add_argument("--aligner", type=str, default="dbalign",
                      help="Aligner to use ('simalign' or 'dbalign').")
  parser.add_argument("--output_file", type=str, default="expandnet_step2_align.out.tsv",
                      help="Output file to save the file with alignments to.")
  
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()

  print(f"Languages:   {args.lang_src} -> {args.lang_tgt}")
  print(f"Aligner:     {args.aligner}")
  print(f"Input file:  {args.translation_df_file}")
  print(f"Output file: {args.output_file}")

  # Initialize aligner in main process (optional, for testing)
  # Worker processes will initialize their own via the align() function
  if args.aligner == 'simalign':
    from simalign import SentenceAligner
    _aligner_obj = SentenceAligner(model="xlmr", layer=8, token_type="bpe", matching_methods="i")
  elif args.aligner == 'dbalign':
    from align_utils import DBAligner
    if args.dict == 'bn':
      print("Initializing DBAlign with BabelNet.")
      _aligner_obj = DBAligner(args.lang_src, args.lang_tgt)
    else:
      print("Initializing DBAlign with Provided Dictionary.")
      _aligner_obj = DBAligner(args.lang_src, args.lang_tgt, 'custom', args.dict)

  from pandarallel import pandarallel

  pandarallel.initialize(progress_bar=True, nb_workers=5)

  print(f"Loading data from {args.translation_df_file}...")
  df_sent = pd.read_csv(args.translation_df_file, sep='\t')
  print(f"Loaded {len(df_sent)} sentences\n")

  print("Aligning sentences...")
  # Create a picklable function with captured parameters using functools.partial
  init_params = {
    'lang_src': args.lang_src,
    'lang_tgt': args.lang_tgt,
    'dict': args.dict
  }
  align_row_func = partial(align_row,
                          lang_src=args.lang_src,
                          lang_tgt=args.lang_tgt,
                          aligner_type_val=args.aligner,
                          init_params=init_params)
  
  df_sent['alignment'] = df_sent.parallel_apply(align_row_func, axis=1)

  print(f"\nSaving results to {args.output_file}...")
  df_sent.to_csv(args.output_file, sep='\t', index=False)
  print("Complete!")
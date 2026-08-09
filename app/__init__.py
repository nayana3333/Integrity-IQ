import os

# nltk >=3.10 installs a MetaPathFinder (CWE-427 mitigation) that blocks any
# import resolving to a path under the current working directory, if nltk is
# anywhere in the call stack (nltk -> textstat -> ... -> regex). That's a
# false positive for us: our virtualenv lives at ./.venv, i.e. *inside* the
# project directory, so nltk's dependency `regex` resolves as a "CWD import"
# and gets blocked. This has to be set before `textstat`/`nltk` are imported
# anywhere, which is why it's here at the top of the top-level package.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

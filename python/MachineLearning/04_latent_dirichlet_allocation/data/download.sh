#!/bin/sh
wget http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
wget http://www.cs.princeton.edu/~blei/lda-c/ap.tgz

tar xzf ap.tgz

python -m gensim.scripts.make_wiki enwiki-latest-pages-articles.xml.bz2 wiki_en_output



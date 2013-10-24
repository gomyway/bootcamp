
class NaiveBayes
  def collect_all_words(self, items):
      all_words = []
      for item in items:
          for w in item.all_words:
              words.append(w)
      return all_words

  def identify_top_words(self, all_words):
      freq_dist = nltk.FreqDist(w.lower() for w in all_words)
      return freq_dist.keys()[:1000]
  
    def read_reuters_metadata(self, cats_file):
      f = open(cats_file, 'r')
      lines = f.readlines()
      f.close()
      return lines
  
    def features(self, top_words):
      word_set = set(self.all_words)
      features = {}
      for w in top_words:
          features["w_%s" % w] = (w in word_set)
      return features
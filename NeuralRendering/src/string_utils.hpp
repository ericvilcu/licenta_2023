#include <string>

inline bool endswith(const std::string& s, const std::string& end) {
	size_t n = s.size(), m = end.size();
	if (m > n)return false;
	size_t d = n - m;
	for (int i = 0; i < m; ++i)if (s[d + i] != end[i])return false;
	return true;
}
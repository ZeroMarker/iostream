#include<iostream>
#include<cmath>
#include<string>
#include<cstring>

bool check(char id[]) {
	int index, sum, num;
	for (sum = index = 0; index < 17; index++)
		sum += ((int)pow(2, 17 - index) % 11) * (id[index] - '0');
	num = (12 - (sum % 11)) % 11;
	if (num < 10)
		return (num == id[17] - '0');
	else
		return (id[17] == 'X');
}
int main() {
    std::string card = "371329200105010671"; 
  
    const int length = card.length(); 
  
    // declaring character array (+1 for null terminator) 
    char* char_array = new char[length + 1]; 
  
    // copying the contents of the 
    // string to char array 
    strcpy(char_array, card.c_str()); 
    std::cout << "Checking" << std::endl;
    std::cout << check(char_array) << std::endl;
    return 0;
}
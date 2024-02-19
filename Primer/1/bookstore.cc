#include <iostream>
#include "Sales_item.h"

using namespace std;

int main(void) {
	Sales_item total; // Varible to store next trans record
	if(cin >> total) {	// ./a.out <./data/book_sales >out
		Sales_item trans; // Store sum
		// Read and process record
		while (cin >> trans) {
			// If we process same book
			if(total.isbn() == trans.isbn())
				total += trans; // Renew sum sale
			else {
				// Print previous book result
				cout << total << endl;
				total = trans; // total = next book sale
			}
			cout << total << endl; // Print last book
		}
	} else {
		cerr << "No Data?!" << endl;
		return -1;
	}
	return 0;
}



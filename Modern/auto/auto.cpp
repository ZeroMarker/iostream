template<typename T, typename U>
auto add2(T x, U y) -> decltype(x+y){
    return x + y;
}
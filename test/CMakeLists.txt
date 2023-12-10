add_executable(pinholePlanesweepTest pinholePlanesweepTest.cpp)
target_link_libraries(pinholePlanesweepTest PSL ${OpenCV_LIBS} ${Boost_LIBRARIES})
add_executable(fisheyePlanesweepTest fisheyePlanesweepTest.cpp)
target_link_libraries(fisheyePlanesweepTest PSL ${OpenCV_LIBS} ${Boost_LIBRARIES})

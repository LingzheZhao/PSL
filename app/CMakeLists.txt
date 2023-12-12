add_executable(cudaPlanesweepMAPTk cudaPlanesweepMAPTk.cpp)
target_link_libraries(cudaPlanesweepMAPTk psl ${VTK_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

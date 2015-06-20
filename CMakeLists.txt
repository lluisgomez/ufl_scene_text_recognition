cmake_minimum_required (VERSION 2.8.7)
project (ufl-scene-text-recognition)


find_package(OpenCV REQUIRED)

add_executable (ufl_predict_char ufl_predict_char.cpp linear.cpp tron.cpp)
add_executable (ufl_predict_word ufl_predict_word.cpp linear.cpp tron.cpp)
add_executable (beam_search beam_search.cpp)


IF(OpenCV_FOUND)
  target_link_libraries (ufl_predict_char ${CMAKE_SOURCE_DIR}/liblinear-1.94/blas/blas.a ${OpenCV_LIBS})
  target_link_libraries (ufl_predict_word ${CMAKE_SOURCE_DIR}/liblinear-1.94/blas/blas.a ${OpenCV_LIBS})
  target_link_libraries (beam_search ${OpenCV_LIBS})
ENDIF()

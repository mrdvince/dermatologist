cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(check_files BASE_DIR FILES_TO_CHECK MISSING_FILES)
    foreach(FILE_TO_CHECK ${${FILES_TO_CHECK}})
        if(EXISTS "${BASE_DIR}/${FILE_TO_CHECK}")
            list(FIND ${FILES_TO_CHECK} ${FILE_TO_CHECK} MD5_IDX)
        else()
            list(APPEND RESULT_FILES ${FILE_TO_CHECK})
        endif()
    endforeach()
    
    set(${MISSING_FILES} ${RESULT_FILES} PARENT_SCOPE)
endfunction()
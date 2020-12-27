function error_if_nan(x)

    if any(isnan(x(:))) 
        error('We have NaNs.') 
    end

end
# # Set prior_CINE:
# # labelmap = 0 is 'bg', 1 is 'lv', 2 is 'myo', 3 is 'rv'
# PRIOR_CINE = {
#     (1,):   (1, 0, 0),
#     (2,):   (1, 1, 0), #Here maybe let open? (1,1,0) or close (1,0,0)
#     (3,):   (1, 0, 0),
#     (1, 2): (1, 0, 0),
#     (1, 3): (2, 0, 0),
#     (2, 3): (1, 1, 0)  #Here maybe let open? (1,1,0) or close (1,0,0)
# }

# # Set prior_LGE: # labelmap = 0 is 'bg', 1 is 'lv'
# PRIOR_EXVIVO = {(1,):   (1, 0, 0)}

# # Set prior_LGE:
# # labelmap = 0 is 'bg', 1 is 'lv', 2 is 'myo', 3 is 'mi'
# PRIOR_LGE = {
#     (1,):   (1, 0, 0),
#     (2,):   (1, 1, 0),
#     (3,):   (1, 0, 0),
#     (1, 2): (1, 0, 0),
#     (1, 3): (1, 0, 0),
#     (2, 3): (1, 1, 0)
# }

# Set prior_CINE:
# labelmap = 0 is 'bg', 1 is 'lv', 2 is 'myo', 3 is 'rv'
PRIOR_CINE = {
    (1,):   (1, 0, 0),
    (2,):   (1, -1, 0), #Here maybe let open? (1,1,0) or close (1,0,0)
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (2, 0, 0),
    (2, 3): (1, -1, 0)  #Here maybe let open? (1,1,0) or close (1,0,0)
}

# Set prior_LGE: # labelmap = 0 is 'bg', 1 is 'lv'
PRIOR_EXVIVO = {(1,):   (1, 0, 0)}

# Set prior_LGE:
# labelmap = 0 is 'bg', 1 is 'lv', 2 is 'myo', 3 is 'mi'
PRIOR_LGE = {
    (1,):   (1, 0, 0),
    (2,):   (1, -1, 0),
    (3,):   (1, 0, 0),
    (1, 2): (1, 0, 0),
    (1, 3): (1, 0, 0),
    (2, 3): (1, 1, 0)
}

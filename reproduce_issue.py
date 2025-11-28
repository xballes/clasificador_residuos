
from waste_classifier import WasteClassifier

def test_classification():
    classifier = WasteClassifier()
    
    # Image 1 (undist_1764003164.png)
    # Object #1 (Previously Carton, Should be Bottle)
    # Features from log:
    # metallic_score = 0.5284178063282021
    # specular_ratio = 0.30633457540694253
    # specular_ratio_top = 0.0
    # gradient_mean = 60.00054107474955
    # is_metallic_color = 0.0
    # is_transparent_color = 0.0
    # is_brown_color = 0.0
    # circularity = 0.8031776224056109
    # elongation_ratio = 1.2025317851327728
    # edge_density_top = 0.05183845690174804
    
    feat_img1_obj1 = {
        "metallic_score": 0.528,
        "specular_ratio": 0.306,
        "specular_ratio_top": 0.0,
        "gradient_mean": 60.0,
        "is_metallic_color": 0.0,
        "circularity": 0.803,
        "elongation_ratio": 1.20,
        "aspect_ratio": 1.0/1.20, # Assuming aspect is inverse of elongation for standing objects roughly
        "edge_density_top": 0.0518
    }
    # Note: aspect_ratio wasn't in logs but is used in code. 
    # Usually aspect = w/h. Elongation = max(w,h)/min(w,h).
    # If standing bottle, h > w. aspect < 1. 
    # If elongation is 1.2, aspect is likely 1/1.2 = 0.83.
    # Code checks: is_square = 0.70 <= aspect <= 1.30. 0.83 is in range.
    feat_img1_obj1["aspect_ratio"] = 0.83

    print("\n--- Image 1 Object #1 (Expected: BOTELLA) ---")
    cls, conf, scores = classifier.classify(feat_img1_obj1)
    classifier.print_classification(cls, conf, scores)
    
    
    # Image 1 Object #3 (Previously Lata, Should be Bottle)
    # metallic_score = 0.7020003633507456
    # circularity = 0.3784291108810578
    # elongation_ratio = 2.3625600585257898
    # is_metallic_color = 1.0
    
    feat_img1_obj3 = {
        "metallic_score": 0.702,
        "is_metallic_color": 1.0,
        "circularity": 0.378,
        "elongation_ratio": 2.36,
        "aspect_ratio": 1.0/2.36, # ~0.42
        "edge_density_top": 0.03, # From log
        "specular_ratio_top": 0.73
    }
    
    print("\n--- Image 1 Object #3 (Expected: BOTELLA) ---")
    cls, conf, scores = classifier.classify(feat_img1_obj3)
    classifier.print_classification(cls, conf, scores)


    # Image 2 Object #1 (Was Bottle, Should stay Bottle)
    # circularity = 0.592514777907062
    # elongation_ratio = 1.2837837837837838
    # edge_density_top = 0.041666666666666664
    
    feat_img2_obj1 = {
        "metallic_score": 0.522,
        "circularity": 0.592,
        "elongation_ratio": 1.28,
        "aspect_ratio": 1.0/1.28, # ~0.78
        "edge_density_top": 0.041,
        "specular_ratio_top": 0.0
    }
    
    print("\n--- Image 2 Object #1 (Expected: BOTELLA) ---")
    cls, conf, scores = classifier.classify(feat_img2_obj1)
    classifier.print_classification(cls, conf, scores)

    # Image 3 (undist_1764003152.png) Object #3 (Was Carton, Should be Bottle)
    # metallic_score = 0.5960161739479479
    # specular_ratio = 0.5049207217058502
    # specular_ratio_top = 0.8625
    # circularity = 0.8213220102841572
    # elongation_ratio = 1.0
    # edge_density_top = 0.04888152444076222
    
    feat_img3_obj3 = {
        "metallic_score": 0.596,
        "specular_ratio": 0.505,
        "specular_ratio_top": 0.8625,
        "gradient_mean": 65.5,
        "is_metallic_color": 0.0,
        "circularity": 0.821,
        "elongation_ratio": 1.0,
        "aspect_ratio": 1.0, 
        "edge_density_top": 0.049
    }
    
    print("\n--- Image 3 Object #3 (Expected: BOTELLA) ---")
    cls, conf, scores = classifier.classify(feat_img3_obj3)
    classifier.print_classification(cls, conf, scores)


if __name__ == "__main__":
    test_classification()
